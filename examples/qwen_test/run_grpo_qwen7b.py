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
import sys
sys.path.append(".")  # Adjust the path to import from the parent directory

import argparse
import os
import pprint
from collections import defaultdict
from typing import Any, Dict

from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset
from mathcl import MathCLDataset
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType, TaskDataSpec
from nemo_rl.distributed.virtual_cluster import init_ray
#from nemo_rl.environments.math_environment import MathEnvironment
from custom_math_environment import CustomMathEnvironment
from nemo_rl.models.generation.interfaces import configure_generation_config
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


# ===============================================================================
#                             Math Data Processor
# ===============================================================================


def mathcl_data_processor(
    datum_dict: Dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary (directly loaded from data/hf_datasets/mathcl.py) into a DatumSpec for the Math Environment."""
    print(f"@SHUANGY@, datum_dict: {datum_dict}")
#    problem = datum_dict["problem"]
#    solution = str(datum_dict["answer"])
#    extra_env_info = {"ground_truth": solution}

    user_message = datum_dict["messages"]
    problem = user_message[0]["content"]
    extra_env_info = {"ground_truth": user_message[1]["content"]}

    message_log: LLMMessageLogType = []

    user_message = {
        "role": "user",
        "content": task_data_spec.prompt.format(problem),
    }
    print(f"@SHUANGY@, user_message: {user_message}")
    message = tokenizer.apply_chat_template(
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    user_message["token_ids"] = tokenizer(message, return_tensors="pt")["input_ids"][0]
    user_message["content"] = message
    message_log.append(user_message)    

    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length > max_seq_length:
        # make smaller and mask out
        for message in message_log:
            message["token_ids"] = message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    output = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": datum_dict["task_name"],
    }
    return output


# Example of a generic math data processor
def math_data_processor(
    datum_dict: Dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary (directly loaded from dataset) into a DatumSpec for the Math Environment."""
    problem = datum_dict["problem"]
    solution = str(datum_dict["ground_truth_answer"])
    extra_env_info = {"ground_truth": solution}

    template = task_data_spec.custom_template
    message_log: LLMMessageLogType = []

    # system prompt
    if task_data_spec.system_prompt:
        sys_message = {"role": "system", "content": task_data_spec.system_prompt}
        message = tokenizer.apply_chat_template(
            [sys_message],
            chat_template=template,
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )
        sys_message["token_ids"] = tokenizer(message, return_tensors="pt")["input_ids"][
            0
        ]
        message_log.append(sys_message)

    # user prompt
    if task_data_spec.prompt:
        problem = task_data_spec.prompt.format(problem)
    user_message = {"role": "user", "content": problem}
    message = tokenizer.apply_chat_template(
        [user_message],
        chat_template=template,
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    user_message["token_ids"] = tokenizer(message, return_tensors="pt")["input_ids"][0]
    user_message["content"] = message
    message_log.append(user_message)

    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length > max_seq_length:
        # make smaller and mask out
        for message in message_log:
            message["token_ids"] = message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    output = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    if "task_name" in datum_dict:
        output["task_name"] = datum_dict["task_name"]
    return output


def setup_data(tokenizer: AutoTokenizer, data_config: DataConfig, env_configs):
    print("\n▶ Setting up data...")
    math_task_spec = TaskDataSpec(
        task_name="math",
        prompt_file=data_config["prompt_file"],
        system_prompt_file=data_config["system_prompt_file"],
    )

    # Load OpenMathInstruct2Dataset using reinforcer datasets
    if data_config["dataset_name"] == "math-cl":
        print(f"Loading pe-nlp/math-cl for training and validation")
        data = MathCLDataset()
    else:
        raise ValueError(f"No processor for dataset {data_config['dataset_name']}.")

    task_data_processors = defaultdict(
        lambda: (math_task_spec, mathcl_data_processor)
    )
    task_data_processors["math"] = (math_task_spec, mathcl_data_processor)

    math_env = CustomMathEnvironment.options(
        runtime_env={
            "py_executable": CustomMathEnvironment.DEFAULT_PY_EXECUTABLE,
            "env_vars": dict(os.environ),  # Pass thru all user environment variables
        }
    ).remote(env_configs["math"])
    dataset = AllTaskProcessedDataset(
        data.formatted_ds["train"],
        tokenizer,
        math_task_spec,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )

    val_dataset = AllTaskProcessedDataset(
        data.formatted_ds["validation"],
        tokenizer,
        math_task_spec,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )

    task_to_env = defaultdict(lambda: math_env)
    task_to_env["math"] = math_env
    return dataset, val_dataset, task_to_env, task_to_env


def main():
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "grpo_qwen_7b.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Print config
    print("Final config:")
    pprint.pprint(config)

    # Get the next experiment directory with incremented ID
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}"
        )

    init_ray()

    # setup tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    # setup data
    dataset, val_dataset, task_to_env, val_task_to_env = setup_data(
        tokenizer, config["data"], config["env"]
    )
    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)
    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        val_task_to_env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )



if __name__ == "__main__":
    main()
