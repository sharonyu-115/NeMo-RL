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

"""Build the mixed math+IF dataset for the multi-teacher MOPD experiment.

Train: 8k DAPO-17k math rows + 8k Nemotron-RL-instruction_following rows,
shuffled. Val: AIME24 (30) + 128 held-out DAPO + 64 held-out IF rows.
Every row carries the agent_ref that routes it to its gym agent server —
which is also the key MOPD uses to pick the teacher.

Inputs: $HF_HOME/nanov3_data/{train,val}-split.jsonl (from
prepare_mopd_gym_data.py) and the gym IF train artifact from HF.
Output: $HF_HOME/mopd_mt_data/{train,val}-split.jsonl
"""

import json
import os
import random
from pathlib import Path

from huggingface_hub import hf_hub_download

MATH_AGENT = {"name": "math_with_judge_simple_agent", "type": "responses_api_agents"}
IF_AGENT = {
    "name": "instruction_following_simple_agent",
    "type": "responses_api_agents",
}

SEED = 42
N_TRAIN_PER_DOMAIN = 8000
N_VAL_IF_HELDOUT = 64

# Required by InstructionFollowingRunRequest (gym IF server app.py).
IF_REQUIRED_KEYS = {"id", "instruction_id_list", "prompt", "kwargs"}

# v3: math val = DAPO's canonical AIME-2024 eval (960 rows = 30 problems x 32
# repeats), formatted identically to DAPO-17k train rows, capped for avg@k.
AIME_REPEATS_PER_PROBLEM = 8


def aime24_dapo_rows(repeats_per_problem: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("BytedTsinghua-SIA/AIME-2024", split="train")
    seen: dict[str, int] = {}
    rows = []
    for example in ds:
        q = example["prompt"][0]["content"]
        if seen.get(q, 0) >= repeats_per_problem:
            continue
        seen[q] = seen.get(q, 0) + 1
        rows.append(
            {
                "agent_ref": MATH_AGENT,
                "responses_create_params": {"input": example["prompt"]},
                "question": q,
                "expected_answer": example["reward_model"]["ground_truth"],
            }
        )
    print(
        f"AIME24(DAPO): {len(seen)} problems x <={repeats_per_problem} repeats "
        f"= {len(rows)} rows"
    )
    return rows


def main() -> None:
    hf_home = os.environ["HF_HOME"]
    src = Path(hf_home) / "nanov3_data"
    out = Path(hf_home) / "mopd_mt_data_v3"
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    # Math: reuse the staged DAPO-17k rows (already carry the math agent_ref).
    math_rows = [json.loads(l) for l in open(src / "train-split.jsonl")]
    assert all(r["agent_ref"]["name"] == MATH_AGENT["name"] for r in math_rows[:5])
    rng.shuffle(math_rows)
    math_train = math_rows[:N_TRAIN_PER_DOMAIN]

    # v3 math val: DAPO-format AIME24 avg@k rows (train/eval format identical).
    aime_val = aime24_dapo_rows(AIME_REPEATS_PER_PROBLEM)

    # IF: the exact artifact the gym IF agent config declares as its train set.
    if_path = hf_hub_download(
        repo_id="nvidia/Nemotron-RL-instruction_following",
        filename="instruction_following.jsonl",
        repo_type="dataset",
    )
    if_rows = []
    for line in open(if_path):
        r = json.loads(line)
        missing = IF_REQUIRED_KEYS - r.keys()
        assert not missing, f"IF row missing required keys: {missing}"
        if_rows.append({"agent_ref": IF_AGENT, **r})
    rng.shuffle(if_rows)
    if_train = if_rows[:N_TRAIN_PER_DOMAIN]
    if_val = if_rows[N_TRAIN_PER_DOMAIN : N_TRAIN_PER_DOMAIN + N_VAL_IF_HELDOUT]

    train = math_train + if_train
    rng.shuffle(train)
    # Fixed val order matters for per-domain analysis: math (AIME avg@k) rows
    # first, then IF. The DAPO held-out slice is dropped in v3 (AIME avg@8
    # provides the sensitivity single-shot AIME lacked).
    val = aime_val + if_val

    with open(out / "train-split.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with open(out / "val-split.jsonl", "w") as f:
        for r in val:
            f.write(json.dumps(r) + "\n")

    print(
        f"train: {len(train)} rows ({len(math_train)} math + {len(if_train)} IF); "
        f"val: {len(val)} rows ({len(aime_val)} AIME24-avg@{AIME_REPEATS_PER_PROBLEM} "
        f"+ {len(if_val)} IF held-out) -> {out}"
    )


if __name__ == "__main__":
    main()
