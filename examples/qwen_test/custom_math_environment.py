# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import contextlib
import io
import logging
import re  # Add the re module for regex
from typing import Dict, List, Optional, Tuple, TypedDict

import ray
import torch
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.environments.metrics import (
    calculate_pass_rate_per_prompt,
)
from nemo_rl.environments.utils import chunk_list_to_workers


class MathEnvConfig(TypedDict):
    num_workers: int
    stop_strings: Optional[List[str]] = None  # Default stop strings for this env


@contextlib.contextmanager
def _mute_output():
    devnull_out, devnull_err = io.StringIO(), io.StringIO()
    with (
        contextlib.redirect_stdout(devnull_out),
        contextlib.redirect_stderr(devnull_err),
    ):
        yield


@ray.remote
class HFVerifyWorker:
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM

    def __init__(self):
        logging.getLogger("math_verify").setLevel(logging.CRITICAL)

        # Use Latex and plain math extraction from predictions
        # https://github.com/huggingface/Math-Verify?tab=readme-ov-file#extraction-targets
        self.verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(
                ExprExtractionConfig(),
                LatexExtractionConfig(),
            ),
        )

    def is_format_correct(self, completion: str) -> bool:
        """Check if the completion has the correct format.
        
        Args:
            completion: str. The completion to check.
            
        Returns:
            bool. True if the format is correct, False otherwise.
        """
        pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
        if not re.match(pattern, completion, re.DOTALL | re.MULTILINE):
            return False
            
        # Check if all tags only appear once
        tags = ["<think>", "</think>", "<answer>", "</answer>"]
        for tag in tags:
            if completion.count(tag) != 1:
                return False
        
        # Check if <think>...</think> is empty
        think_pattern = r"<think>(.*?)</think>"
        think_match = re.search(think_pattern, completion, re.DOTALL | re.MULTILINE)
        if think_match and think_match.group(1).strip() == "":
            return False
        
        return True
    
    def calculate_format_reward(self, completion: str) -> float:
        """Calculate reward based on format correctness.
        
        Args:
            completion: str. The completion to check.
            
        Returns:
            float. 1.0 if the format is correct, 0.0 otherwise.
        """
        return 1.0 if self.is_format_correct(completion) else 0.0
        
    def extract_answer(self, completion: str) -> str:
        """Extract the answer from a correctly formatted completion.
        
        Args:
            completion: str. The correctly formatted completion.
            
        Returns:
            str. The extracted answer content or the original completion if format is incorrect.
        """
        if not self.is_format_correct(completion):
            return completion
            
        answer_pattern = r"<answer>(.*?)</answer>"
        answer_match = re.search(answer_pattern, completion, re.DOTALL | re.MULTILINE)
        if answer_match:
            return answer_match.group(1).strip()
        return completion

    def verify(
        self, pred_responses: List[str], ground_truths: List[str]
    ) -> List[float]:
        """Verify the correctness of the predicted responses against the ground truth.

        Args:
            pred_responses: List[str]. The predicted responses from the LLM.
            ground_truths: List[str]. The ground truth responses.

        Returns:
            List[float]. The rewards for each predicted response.
        """
        results = []
        
        # Debug print for a sample of the records (first 2 items or all if less than 2)
        sample_size = min(5, len(pred_responses))
        print(f"\n==== Debug Info: Verifying {len(pred_responses)} responses ====")
        
        for i, (response, ground_truth) in enumerate(zip(pred_responses, ground_truths)):
            try:
                # Show debug info for the sample
                is_sample = i < sample_size
                if is_sample:
                    print(f"\n--- Record {i} ---")
                    print(f"Ground truth: {ground_truth}")
                    print(f"Response format correct: {self.is_format_correct(response)}")
                
                # First, check if the format is correct
                if not self.is_format_correct(response):
                    if is_sample:
                        print(f"Format incorrect. Reward: 0.0")
                        print(f"Response: {response}")
                    results.append(0.0)  # Return 0 for incorrect format
                    continue
                    
                # Extract the answer part for validation
                extracted_answer = self.extract_answer(response)
                if is_sample:
                    print(f"Extracted answer: {extracted_answer}")
                
                ground_truth_parsable = "\\boxed{" + ground_truth + "}"
                if is_sample:
                    print(f"Parsable ground truth: {ground_truth_parsable}")
                
                with _mute_output():
                    try:
                        ret_score, _ = self.verify_func(
                            [ground_truth_parsable], [extracted_answer]
                        )
                        # Convert to binary reward: 1.0 if correct, 0.0 if incorrect
                        ret_score = 1.0 if ret_score > 0.0 else 0.0
                    except Exception as e:
                        if is_sample:
                            print(f"Exception during verification: {str(e)}")
                        ret_score = 0.0

                if is_sample:
                    print(f"Reward: {ret_score}")
                
                results.append(float(ret_score))
            except Exception as e:
                if is_sample:
                    print(f"Exception during processing: {str(e)}")
                results.append(0.0)
        
        if sample_size > 0:
            print(f"\n==== End Debug Info ====\n")
        
        return results


class MathEnvironmentMetadata(TypedDict):
    ground_truth: str


@ray.remote
class CustomMathEnvironment(EnvironmentInterface):
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM

    def __init__(self, cfg: MathEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        self.workers = [
            HFVerifyWorker.options(
                runtime_env={"py_executable": HFVerifyWorker.DEFAULT_PY_EXECUTABLE}
            ).remote()
            for _ in range(self.num_workers)
        ]

    def shutdown(self):
        # shutdown all workers
        for worker in self.workers:
            ray.kill(worker)

    def step(
        self,
        message_log_batch: List[List[Dict[str, str]]],
        metadata: List[MathEnvironmentMetadata],
    ) -> EnvironmentReturn:
        """Runs a step in the math environment.

        Args:
            message_log: List[List[Dict[str, str]]]. A batch of OpenAI-API-like message logs that represent interactions with the LLM.
            metadata: List[MathEnvironmentMetadata]. The grader will use the 'ground_truth' key to evaluate correctness.

        Returns:
            EnvironmentReturn: A tuple containing:
                - List[Dict[str, str]]: Observations/responses batch
                - List[Dict]: Updated metadata
                - List[str]: Next stop strings for the next turn
                - Tensor: Rewards tensor
                - Tensor: Done flags tensor
        """
        # Extract the assistant's responses from the message history
        # Each message list should have at least one assistant response
        assistant_response_batch = []
        for conversation in message_log_batch:
            assistant_responses = [
                interaction["content"]
                for interaction in conversation
                if interaction["role"] == "assistant"
            ]
            assistant_response_batch.append("".join(assistant_responses))

        ground_truths = [g["ground_truth"] for g in metadata]

        chunked_assistant_response_batch = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )
        chunked_ground_truths = chunk_list_to_workers(ground_truths, self.num_workers)

        # # Process each chunk in parallel
        futures = [
            self.workers[i].verify.remote(chunk, ground_truth_chunk)
            for i, (chunk, ground_truth_chunk) in enumerate(
                zip(chunked_assistant_response_batch, chunked_ground_truths)
            )
        ]

        results = ray.get(futures)

        # flatten the results
        results = [item for sublist in results for item in sublist]
        
        # We need to store format correctness info for observations
        format_check_futures = [
            self.workers[i % self.num_workers].is_format_correct.remote(response)
            for i, response in enumerate(assistant_response_batch)
        ]
        format_checks = ray.get(format_check_futures)
        
        observations = [
            {
                "role": "environment",
                "content": "Environment: format error" if not is_format_correct else
                           "Environment: correct" if result > 0.0 else
                           "Environment: incorrect",
            }
            for result, is_format_correct in zip(results, format_checks)
        ]
        
        # Create a tensor of rewards and done flags
        rewards = torch.tensor(results).cpu()
        done = torch.ones_like(rewards).cpu()

        next_stop_strings = [None] * len(message_log_batch)

        # Include format check info in metadata
        for i, meta in enumerate(metadata):
            meta["format_error"] = not format_checks[i]  # True if format is incorrect

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards,
            terminateds=done,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> Tuple[BatchedDataDict, dict]:
        """Computes metrics for this environment given a global rollout batch.

        Every rank will run this function, so you're free to use distributed
        calculations if you'd prefer for heavy metrics.
        """
        # Check if we have format correctness info stored in metadata
        format_errors = []
        if "metadata" in batch:
            for meta in batch["metadata"]:
                if meta and "format_error" in meta:
                    format_errors.append(meta["format_error"])
        
        batch["rewards"] = (
            batch["rewards"] * batch["is_end"]
        )  # set a reward of 0 for any incorrectly ended sequences
        if (batch["rewards"] == 1).float().sum() > 0:
            correct_solution_generation_lengths = (
                (batch["generation_lengths"] - batch["prompt_lengths"])[
                    batch["rewards"] == 1
                ]
                .float()
                .mean()
                .item()
            )
        else:
            correct_solution_generation_lengths = 0

        # Calculate format error rate if we have format error data
        format_error_rate = sum(format_errors) / len(format_errors) if format_errors else 0.0

        metrics = {
            # "table": table, TODO @sahilj WIP
            "accuracy": batch["rewards"].mean().item(),
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(
                batch["text"], batch["rewards"]
            ),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "correct_solution_generation_lengths": correct_solution_generation_lengths,
            "format_error_rate": format_error_rate,
        }
        print(f"Metrics: {metrics}")
        return batch, metrics
