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

from typing import Optional
from datasets import load_dataset
from nemo_rl.data.interfaces import TaskDataSpec


def format_math(data, output_key: str = "answer"):
    return {
        "messages": [
            {
                "role": "user",
                "content": data["problem"],
            },
            {
                "role": "assistant",
                "content": data[output_key],
            },
        ],
        "task_name": "math",
    }


def prepare_mathcl_dataset():
    """Load and split the OpenMathInstruct-2 dataset into train and validation sets using HF's train_test_split."""
    print(
        f"WARNING: For reproducible experiments, preprocess the dataset once and define your own HfDataset subclass that directly uses the preprocessed datasets."
    )

    # Load the original dataset
    train_set = load_dataset("pe-nlp/math-cl", split="train")
    # rename ground_truth_answer to answer
    train_set = train_set.rename_column("ground_truth_answer", "answer")

    # Format the examples, removing original columns
    train_formatted = train_set.map(
        format_math, remove_columns=train_set.column_names
    )

    math_500 = load_dataset("HuggingFaceH4/MATH-500", split="test")

    val_formatted = math_500.map(
        format_math, remove_columns=math_500.column_names
    )

    return {
        "train": train_formatted,
        "validation": val_formatted,
    }


class MathCLDataset:
    def __init__(
        self,
        seed: int = 42,
        prompt_file: str = None,
    ):
        """Initialize the MathCL dataset.

        Args:
            seed: Random seed for reproducible behavior
            prompt_file: Path to prompt template file
        """
        self.formatted_ds = prepare_mathcl_dataset()
        
        self.task_spec = TaskDataSpec(
            task_name="math-cl",
            prompt_file=prompt_file,
        )
