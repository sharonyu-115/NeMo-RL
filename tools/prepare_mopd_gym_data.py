# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Stage assets for the MOPD smoke recipe (mopd-qwen3-1.7b-3n8g-megatron-pack).

Generates NeMo Gym math jsonl splits at $HF_HOME/nanov3_data/{train,val}-split.jsonl
(the paths the recipe expects) and downloads the Qwen/Qwen3-1.7B checkpoint into
the HF cache. Row format mirrors the Gym repo's
resources_servers/math_with_judge/prepare_{dapo17k,aime24}.py scripts.
"""

import json
import os
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download

SYSTEM_PROMPT = (
    "Your task is to solve a math problem.  Make sure to put the answer "
    "(and only the answer) inside \\boxed{}."
)

# Route every row to the math agent server from math_with_judge.yaml. Without
# agent_ref, NeMo Gym rollout collection cannot route the row (older nemo_gym
# builds hang with a swallowed KeyError instead of failing fast), and MOPD's
# teacher routing also keys off this name (unmapped names fall back to
# default_teacher_alias).
AGENT_REF = {"name": "math_with_judge_simple_agent", "type": "responses_api_agents"}


def write_train(out_path: Path) -> int:
    ds = load_dataset("YouJiacheng/DAPO-Math-17k-dedup", split="train")
    n = 0
    with open(out_path, "w") as f:
        for example in ds:
            row = {
                "agent_ref": AGENT_REF,
                "responses_create_params": {"input": example["prompt"]},
                "question": example["prompt"][0]["content"],
                "expected_answer": example["reward_model"]["ground_truth"],
            }
            f.write(json.dumps(row) + "\n")
            n += 1
    return n


def write_val(out_path: Path) -> int:
    ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    n = 0
    with open(out_path, "w") as f:
        for example in ds:
            row = {
                "agent_ref": AGENT_REF,
                "responses_create_params": {
                    "input": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": example["problem"]},
                    ]
                },
                "question": example["problem"],
                "expected_answer": example["answer"],
            }
            f.write(json.dumps(row) + "\n")
            n += 1
    return n


def main() -> None:
    hf_home = os.environ.get("HF_HOME")
    assert hf_home, "HF_HOME must be set"
    out_dir = Path(hf_home) / "nanov3_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_train = write_train(out_dir / "train-split.jsonl")
    n_val = write_val(out_dir / "val-split.jsonl")
    print(f"wrote {n_train} train rows, {n_val} val rows to {out_dir}")

    path = snapshot_download("Qwen/Qwen3-1.7B")
    print(f"Qwen/Qwen3-1.7B cached at {path}")


if __name__ == "__main__":
    main()
