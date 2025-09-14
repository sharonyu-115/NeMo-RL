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
import os
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any, NotRequired, Optional, TypedDict, TypeVar, cast

import numpy as np
import ray
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.algorithms.loss_functions import (
    ClippedPGLossConfig,
    ClippedPGLossDataDict,
    ClippedPGLossFn,
)
from nemo_rl.algorithms.utils import calculate_baseline_and_std_per_prompt, set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset, rl_collate_fn
from nemo_rl.data.interfaces import (
    DatumSpec,
)
from nemo_rl.data.llm_message_utils import (
    batched_message_log_to_flat_message,
    get_keys_from_message_log,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import (
    ClusterConfig,
    RayVirtualCluster,
)
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
)
from nemo_rl.experience.rollouts import (
    run_async_multi_turn_rollout,
    run_multi_turn_rollout,
)
from nemo_rl.models.generation.interfaces import (
    GenerationInterface,
)
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import (
    Logger,
    LoggerConfig,
    print_message_log_samples,
)
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.timer import TimeoutChecker, Timer

# ===============================================================================
# Configuration
# ===============================================================================
TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


class GRPOConfig(TypedDict):
    num_prompts_per_step: int
    num_generations_per_prompt: int
    max_num_steps: int
    max_rollout_turns: int
    normalize_rewards: bool
    use_leave_one_out_baseline: bool
    val_period: int
    val_batch_size: int
    val_at_start: bool
    max_val_samples: int
    seed: int


class GRPOSaveState(TypedDict):
    step: int
    val_reward: NotRequired[
        float
    ]  # Optional field - may not be present during training
    consumed_samples: int


def _default_grpo_save_state() -> GRPOSaveState:
    return {
        "step": 0,
        "val_reward": -99999999.0,
        "consumed_samples": 0,
    }


class GRPOLoggerConfig(LoggerConfig):
    num_val_samples_to_print: int  # number of val samples to print to stdout


class MasterConfig(TypedDict):
    policy: PolicyConfig
    loss_fn: ClippedPGLossConfig
    env: dict[str, Any]
    data: DataConfig
    grpo: GRPOConfig
    logger: GRPOLoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig


# ===============================================================================
# Setup & Initialization
# ===============================================================================

# Function to check if KV cache scales should be calculated and synchronized during refit
# TODO: Where and how to calcualte this kv cache scales? 
# TODO: This should be checked only once and reused during the whole training process. Should the flag be stored somewhere?
def _should_sync_kv_scales(master_config: MasterConfig) -> bool:
    """
    Check if KV cache scales should be synchronized during refit.
    
    Returns True if:
    - vLLM backend is used for generation
    - Either kv_cache_dtype is fp8 OR vLLM precision is fp8 (which implies fp8 kv cache)
    - This indicates we need to sync _k_scale and _v_scale values
    """
    generation_config = master_config["policy"]["generation"]
    if generation_config is None:
        return False
    
    backend = generation_config.get("backend", "")
    if backend != "vllm":
        return False
    
    vllm_cfg = generation_config.get("vllm_cfg", {})
    kv_cache_dtype = vllm_cfg.get("kv_cache_dtype", "auto")
    vllm_precision = vllm_cfg.get("precision", "auto")
    
    # Check if either kv_cache_dtype is explicitly fp8 or vLLM precision is fp8
    # When vLLM precision is fp8, it typically implies fp8 kv cache as well
    # should enable kv scale sync when both are true
    return kv_cache_dtype == "fp8" and vllm_precision == "fp8"


def setup(
    master_config: MasterConfig,
    tokenizer: TokenizerType,
    dataset: AllTaskProcessedDataset,
    val_dataset: Optional[AllTaskProcessedDataset],
    processor: Optional[AutoProcessor] = None,
) -> tuple[
    ColocatablePolicyInterface,
    Optional[GenerationInterface],
    tuple[RayVirtualCluster, RayVirtualCluster],
    StatefulDataLoader,
    Optional[StatefulDataLoader],
    ClippedPGLossFn,
    Logger,
    CheckpointManager,
    GRPOSaveState,
    MasterConfig,
]:
    """Main entry point for running GRPO algorithm.

    Returns:
        tuple of policy, cluster, dataloader, tokenizer, loss_fn, math_env, logger, master_config, val_dataloader
    """
    # Extract individual configs for easier access
    policy_config = master_config["policy"]
    generation_config = master_config["policy"]["generation"]
    loss_config = master_config["loss_fn"]
    grpo_config = master_config["grpo"]
    data_config = master_config["data"]
    logger_config = master_config["logger"]
    cluster_config = master_config["cluster"]

    assert generation_config is not None, (
        "A generation config in the PolicyConfig is required for GRPO"
    )

    # Set seed for all random number generators
    set_seed(grpo_config["seed"])

    # ==========================
    #         Logger
    # ==========================
    logger = Logger(logger_config)
    logger.log_hyperparams(master_config)

    # ==========================
    #      Checkpointing
    # ==========================
    checkpointer = CheckpointManager(master_config["checkpointing"])
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    grpo_save_state: Optional[GRPOSaveState] = cast(
        Optional[GRPOSaveState], checkpointer.load_training_info(last_checkpoint_path)
    )
    if grpo_save_state is None:
        grpo_save_state = _default_grpo_save_state()

    # ==========================
    #           Data
    # ==========================
    dataloader = StatefulDataLoader(
        dataset,
        batch_size=grpo_config["num_prompts_per_step"],
        shuffle=data_config["shuffle"],
        collate_fn=rl_collate_fn,
        drop_last=True,
    )
    if last_checkpoint_path is not None:
        dataloader_state_dict = torch.load(
            os.path.join(last_checkpoint_path, "train_dataloader.pt")
        )
        dataloader.load_state_dict(dataloader_state_dict)

    print(f"  ✓ Training dataloader loaded with {len(dataset)} samples")

    # Load validation dataset if provided
    val_dataloader: Optional[StatefulDataLoader] = None
    # If validation is enabled, load the validation dataloader
    if grpo_config["val_period"] > 0 or grpo_config["val_at_start"]:
        assert val_dataset is not None, (
            "Validation dataset is required if validation is enabled"
        )
        val_dataloader = StatefulDataLoader(
            val_dataset,
            batch_size=grpo_config["val_batch_size"],
            shuffle=False,
            collate_fn=rl_collate_fn,
        )
        print(f"  ✓ Validation dataloader loaded with {len(val_dataset)} samples")

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...")
    colocated_inference = generation_config["colocated"]["enabled"]

    if colocated_inference:
        cluster = RayVirtualCluster(
            name="grpo_policy_cluster",
            bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
            * cluster_config["num_nodes"],
            use_gpus=True,
            num_gpus_per_node=cluster_config["gpus_per_node"],
            max_colocated_worker_groups=1
            if generation_config["backend"] == "megatron"
            else 2,
        )
        train_cluster = cluster
        inference_cluster = cluster
        print(f"  ✓ Ray cluster initialized with {cluster_config['num_nodes']} nodes")

    else:
        assert generation_config["backend"] != "megatron", (
            "Non-colocated inference is not supported for Megatron generation backends. "
            "Please use vLLM backend for generation."
        )

        # train resources will be updated through overall and inference resources below
        train_gpus_per_node = cluster_config["gpus_per_node"]
        train_nodes = cluster_config["num_nodes"]

        inference_resources = generation_config["colocated"]["resources"]
        inference_gpus_per_node = inference_resources["gpus_per_node"]
        inference_nodes = inference_resources["num_nodes"]

        # validate and configure resources
        if cluster_config["num_nodes"] == 1:
            assert inference_gpus_per_node > 0, (
                "policy.generation.colocated.resources.gpus_per_node must be > 0 "
                "when cluster.num_nodes = 1 and inference is non-colocated, "
                f"but got {inference_gpus_per_node}."
            )
            assert inference_nodes is None or inference_nodes == 1, (
                "policy.generation.colocated.resources.num_nodes must be 1 or set to null "
                "when cluster.num_nodes = 1 and inference is non-colocated, "
                f"but got {inference_nodes}."
            )
            inference_nodes = 1
            train_gpus_per_node -= inference_gpus_per_node
        else:
            assert inference_nodes > 0, (
                "policy.generation.colocated.resources.num_nodes must be > 0 "
                "when cluster.num_nodes > 1 and inference is non-colocated, "
                f"but got {inference_nodes}."
            )
            assert (
                inference_gpus_per_node is None
                or inference_gpus_per_node == cluster_config["gpus_per_node"]
            ), (
                "policy.generation.colocated.resources.gpus_per_node must be equal to cluster.gpus_per_node or set to null "
                "when cluster.num_nodes > 1 and inference is non-colocated, "
                f"but got {inference_gpus_per_node}."
            )
            inference_gpus_per_node = cluster_config["gpus_per_node"]
            train_nodes -= inference_nodes

        # initialize train cluster
        train_cluster = RayVirtualCluster(
            name="grpo_train_cluster",
            bundle_ct_per_node_list=[train_gpus_per_node] * train_nodes,
            use_gpus=True,
            num_gpus_per_node=train_gpus_per_node,
            max_colocated_worker_groups=1,
        )
        print(
            f"  ✓ Ray train cluster initialized with {train_nodes} nodes with {train_gpus_per_node} GPUs per node"
        )

        # initialize inference cluster
        inference_cluster = RayVirtualCluster(
            name="grpo_inference_cluster",
            bundle_ct_per_node_list=[inference_gpus_per_node] * inference_nodes,
            use_gpus=True,
            num_gpus_per_node=inference_gpus_per_node,
            max_colocated_worker_groups=1,
        )
        print(
            f"  ✓ Ray inference cluster initialized with {inference_nodes} nodes with {inference_gpus_per_node} GPUs per node"
        )

    # ==========================
    #   Training and Inference
    # ==========================
    print("\n▶ Setting up model and training...")

    # vllm model loading prefers clean environment, initialize policy_generation before policy (#52 will fix this)
    backend = generation_config["backend"]
    generation_config["model_name"] = policy_config["model_name"]  # Needed for vLLM

    if backend == "megatron":
        policy_generation = None
        print(
            f"  ✓ Using {backend} backend for generation with {policy_config['model_name']}"
        )
    elif backend == "vllm":
        generation_config = cast(VllmConfig, generation_config)
        if generation_config["vllm_cfg"]["precision"] == "fp8":
            assert loss_config["use_importance_sampling_correction"] is True, (
                "Importance sampling must be enabled for vLLM FP8 generation for good convergence!"
            )

        policy_generation = VllmGeneration(
            cluster=inference_cluster, config=generation_config
        )
        # Worker groups are not initialized until the first call to run something on workergroups.
        # vllm 0.8 fails in initialization if its called in the first training step since it has no clean view of the GPU memory (HF is sharing the same memory).
        policy_generation.finish_generation()
        print(
            f"  ✓ Using vLLM backend for generation with {policy_config['model_name']}"
        )

    if last_checkpoint_path:
        weights_path = Path(last_checkpoint_path) / "policy" / "weights"
        optimizer_path = Path(last_checkpoint_path) / "policy" / "optimizer"
    else:
        weights_path = None
        optimizer_path = None

    policy = Policy(
        cluster=train_cluster,
        config=policy_config,
        tokenizer=tokenizer,
        processor=processor,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=True,
    )

    # if it is not colocated inference, initialize collective communication for update weights
    if not colocated_inference:
        ip, port = train_cluster.get_master_address_and_port()
        print(f"Using ip: {ip}, port: {port} for collective communication")
        # inference cluster + head node of the train cluster
        world_size = inference_nodes * inference_gpus_per_node + 1
        # init collective
        futures_train = policy.init_collective(ip, port, world_size)
        futures_inference = policy_generation.init_collective(ip, port, world_size)  # type: ignore
        # wait for all futures to complete
        ray.get(futures_train + futures_inference)

    # prepare refit info
    state_dict_info = policy.prepare_refit_info()
    policy_generation.prepare_refit_info(state_dict_info)

    loss_fn = ClippedPGLossFn(loss_config)

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n")

    return (
        policy,
        policy_generation,
        (train_cluster, inference_cluster),
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_save_state,
        master_config,
    )


# ===============================================================================
# Core Algorithm Functions
# ===============================================================================


def _should_use_async_rollouts(master_config: MasterConfig) -> bool:
    """Determine if async rollouts should be used based on the configuration.

    Returns True if vLLM backend is used with async_engine enabled.
    """
    generation_config = master_config["policy"]["generation"]
    if generation_config is None:
        return False

    backend = generation_config.get("backend", "")
    if backend != "vllm":
        return False

    vllm_cfg = generation_config.get("vllm_cfg", {})
    return vllm_cfg.get("async_engine", False)

# Need a function to compute the kv cache scales for all the attention layers with the updated policy model
# TODO: Determine the inputs and outputs. inputs: the trained policy model? training data? 
# TODO: How to do the calculation? Caculating the kv cache scales needs to do a foward path with some training data, get the activations of each attention layer and compute the scales based on the activations. 
# TODO: The calcuation needs to be done only when the sync_kv_scales flat is True, and after policy model is updated. 
# TODO: The output should be a dictionary of the kv cache scales for all the attention layers? The structure should be consistent with the required format that can be loaded by vllm using model_runner.model.load_weights() as the other weights. 
#       Code snippet reference: /lustre/fsw/portfolios/coreai/users/shuangy/src/vllm/vllm/model_executor/layers/quantization/kv_cache.py:
#        def create_weights(self, layer: torch.nn.Module):
#            """
#            Create "weight" (aka q_scale, k_scale and v_scale)
#            for an attention layer.
#            """
            # Initialize the Q and KV cache scales to -1.0, an invalid value.
            # If the q and k/v_scales appear in the checkpoint, it will be
            # overwritten when loading weights.
#            layer.q_scale = torch.nn.Parameter(torch.tensor(-1.0),
#                                            requires_grad=False)
#            layer.k_scale = torch.nn.Parameter(torch.tensor(-1.0),
#                                            requires_grad=False)
#            layer.v_scale = torch.nn.Parameter(torch.tensor(-1.0),
#                                            requires_grad=False)
            # Initialize P = softmax(QK^T) scales
#            layer.prob_scale = torch.nn.Parameter(torch.tensor(-1.0),
#                                                   requires_grad=False)
#                                                requires_grad=False)
# TODO: How to pass the kv scales to refit_policy_generation()? refit_policy_generation() is the function that updates the weights of the policy generation interface.
# When refit_policy_generation() invokes update_weights_from_ipc_handles() or update_weights_from_collective(), if it is fp8 and sync_kv_scales is True, the kv scales should be passed to the policy generation interface. load_weights() once invoked will load the kv scales. 
# In order for vllm to really load the kv scales, the kv scales should be passed to the policy generation interface in the same format as the other weights. 
# Additionally, vllm process_weights_after_loading() will be invoked after load_weights() to copy the kv scales to the _k_scale and _v_scale attributes. Reference code: /lustre/fsw/portfolios/coreai/users/shuangy/src/vllm/vllm/model_executor/layers/quantization/kv_cache.py


def compute_kv_scales_with_data(
    policy: ColocatablePolicyInterface,
    sample_data: BatchedDataDict,
    master_config: MasterConfig,
    max_samples: int = 32,
) -> dict[str, float]:
    """
    Compute KV cache scales for all attention layers using calibration data.
    
    Args:
        policy: The policy model to calibrate
        sample_data: Calibration data batch
        master_config: Configuration containing model settings
        max_samples: Maximum number of samples to use for calibration
        
    Returns:
        Dictionary mapping parameter names to scale values for K/V cache quantization
    """
    # TODO: Review the implementation of this function.
    print(f"[KV_SCALES] Computing KV cache scales with {min(max_samples, sample_data.size)} samples...")
    
    # Limit the number of samples for calibration
    if sample_data.size > max_samples:
        sample_data = sample_data.slice(0, max_samples)
    
    # Convert to input format expected by policy
    import torch
    from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
    
    try:
        # Extract tokenized inputs from the batch
        batched_flat, input_lengths = batched_message_log_to_flat_message(
            sample_data["message_log"], 
            pad_value_dict={"token_ids": 0}  # Use 0 as pad token for calibration
        )
        input_ids = batched_flat["token_ids"]
        
        # Convert to tensor if needed
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # For distributed policy, we'll use a simplified approach
        # TODO: Implement proper distributed calibration through worker_group
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = input_ids.to(device)
        
        print(f"[KV_SCALES] Calibration input shape: {input_ids.shape}")
        
        # Skip the complex hook-based calibration for distributed policy
        # TODO: Implement proper distributed calibration using policy.worker_group
        
        # For distributed Policy, we cannot directly access model.named_modules()
        # Instead, we'll use a simplified approach with default scales
        print("[KV_SCALES] Using simplified calibration for distributed policy")
        
        # TODO: For a quick prototype, use a pseudo default scales. 
        #       Need to update later to would use worker_group to run calibration? 
        default_k_scale = 0.1  # Conservative scale for K projections  
        default_v_scale = 0.1  # Conservative scale for V projections
        # TODO: Current use Qwen3-8B-Base as an example, should be obtained from model config
        num_layers = 36  # Default number of layers - should be obtained from model config
        
        # Generate default KV scales for distributed policy
        kv_scales = {}
        print("[KV_SCALES] Generating default KV scales for distributed policy")
        
        # Generate scales for typical transformer layers
        for layer_idx in range(num_layers):
            k_param_name = f"model.layers.{layer_idx}.self_attn.k_scale"
            v_param_name = f"model.layers.{layer_idx}.self_attn.v_scale"
            
            kv_scales[k_param_name] = default_k_scale
            kv_scales[v_param_name] = default_v_scale
        
        print(f"[KV_SCALES] Computed {len(kv_scales)} KV cache scales")
        return kv_scales
    
    except Exception as e:
        print(f"[KV_SCALES] Error computing KV scales: {e}")
        # For training stability, we can either:
        # 1. Re-raise the exception to fail fast and debug issues early
        # 2. Fall back to default scales to continue training
        # Current choice: fallback for robustness, but log the error clearly
        print("[KV_SCALES] Falling back to default scales to maintain training stability")
        print("[KV_SCALES] Note: This may impact FP8 quantization quality")
        
        # Return default scales
        default_scales = {}
        for name, module in policy.model.named_modules():
            if "self_attn" in name:
                default_scales[f"{name}.k_scale"] = 1.0
                default_scales[f"{name}.v_scale"] = 1.0
        return default_scales

def refit_policy_generation(
    policy: ColocatablePolicyInterface,
    policy_generation: GenerationInterface,
    colocated_inference: bool,
    _refit_buffer_size_gb: Optional[int] = None,
    timer: Optional[Timer] = None,
    kv_scales: Optional[dict[str, float]] = None,
) -> None:
    """Refit the policy generation interface with the latest policy weights.

    Args:
        policy: The policy to provide weights to the inference engine.
        policy_generation: The inference engine to refit.
        _refit_buffer_size_gb: The size of the buffer to use for refitting.
            If it is None, the buffer size will be computed by the remaining memory.
            This parameter is primarily used for testing.
        kv_scales: Optional dictionary of KV cache scales for FP8 quantization.
    """
    if colocated_inference:
        policy.offload_before_refit()
        policy_generation.prepare_for_generation(tags=["weights"])

    # Create a context manager that does nothing when timer is None
    timer_context = (
        timer.time("prepare_for_generation/transfer_and_update_weights")
        if timer is not None
        else nullcontext()
    )
    with timer_context:
        # update weights
        update_success = False
        if colocated_inference:
            # get model param keys, which is grouped by size
            grouped_param_keys = policy.prepare_weights_for_ipc(
                _refit_buffer_size_gb=_refit_buffer_size_gb
            )
            total_num_keys = sum(len(k) for k in grouped_param_keys)
            print(
                f"[KV_SCALES] Refit: Split {total_num_keys} keys into {len(grouped_param_keys)} groups"
            )
            
            # Add KV scales to the first group if available
            if kv_scales:
                print(f"[KV_SCALES] Refit: Adding {len(kv_scales)} KV scales to weight update")
                print(f"[KV_SCALES] Refit: KV scales: {kv_scales}")
                # Note: KV scales will be handled by the generation interface
                # They need to be passed separately since they're not regular model weights
            
            # do update
            for keys in grouped_param_keys:
                ipc_handles = policy.get_weights_ipc_handles(keys)
                # Pass KV scales with the first batch only
                if kv_scales and keys == grouped_param_keys[0]:
                    print(f"[KV_SCALES] Refit: Adding {len(kv_scales)} KV scales to weight update via ipc_handles")
                    update_success = policy_generation.update_weights_from_ipc_handles(
                        ipc_handles, kv_scales=kv_scales
                    )
                else:
                    update_success = policy_generation.update_weights_from_ipc_handles(
                        ipc_handles
                    )
                if not update_success:
                    break
        else:
            # update weights through nccl
            futures_train = policy.broadcast_weights_for_collective()
            if kv_scales:
                print(f"[KV_SCALES] Refit: Adding {len(kv_scales)} KV scales to collective weight update")
                futures_inference = policy_generation.update_weights_from_collective(kv_scales=kv_scales)
            else:
                futures_inference = policy_generation.update_weights_from_collective()
            # wait for all futures to complete
            ray.get(futures_train)
            results = ray.get(futures_inference)
            update_success = all(result for result in results if result is not None)

        # check if update is successful
        if not update_success:
            error_tag = "cuda-ipc" if colocated_inference else "nccl"
            error_message = (
                "❌ Error: Updating weights for the generation policy failed during refit.\n"
                f"This often indicates an issue with {error_tag} or "
                "a problem within the generation backend (e.g., vLLM worker).\n"
            )
            raise RuntimeError(error_message)

    if colocated_inference:
        policy.offload_after_refit()
        policy_generation.prepare_for_generation(tags=["kv_cache"])


# ===============================================================================
# Training & Validation
# ===============================================================================


def grpo_train(
    policy: ColocatablePolicyInterface,
    policy_generation: Optional[GenerationInterface],
    dataloader: StatefulDataLoader,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer: TokenizerType,
    loss_fn: LossFunction,
    task_to_env: dict[str, EnvironmentInterface],
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    logger: Logger,
    checkpointer: CheckpointManager,
    grpo_save_state: GRPOSaveState,
    master_config: MasterConfig,
    processor: Optional[AutoProcessor] = None,
) -> None:
    """Run GRPO training algorithm."""
    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config["checkpointing"]["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    # Check if we need to sync KV cache scales (infer from config)
    sync_kv_scales = _should_sync_kv_scales(master_config)
    kv_scales_cache = None  # Cache computed KV scales for reuse
    
    if sync_kv_scales:
        generation_config = master_config["policy"]["generation"]
        vllm_cfg = generation_config.get("vllm_cfg", {})
        backend = generation_config.get("backend", "")
        kv_cache_dtype = vllm_cfg.get("kv_cache_dtype", "auto")
        vllm_precision = vllm_cfg.get("precision", "auto")
        policy_backend = "megatron" if master_config["policy"].get("megatron_cfg", {}).get("enabled", False) else "dtensor"
        
        print(f"[KV_SCALES] FP8 KV cache detected, will sync _k_scale and _v_scale during refit")
        print(f"[KV_SCALES] Configuration: policy_backend={policy_backend}, generation_backend={backend}")
        print(f"[KV_SCALES] vLLM settings: precision={vllm_precision}, kv_cache_dtype={kv_cache_dtype}")
    else:
        print("[KV_SCALES] KV cache scale sync not needed (non-FP8 mode)")

    NEED_REFIT = True
    # If policy_generation is None, use the policy as the generation interface (megatron framework backend)
    if policy_generation is None:
        policy_generation = policy  # type: ignore
        NEED_REFIT = False
    POLICY_GENERATION_STALE = True  # tracks if generation needs a refit before running
    assert policy_generation is not None  # for mypy type check

    # common config/state itmes
    step = grpo_save_state["step"]
    consumed_samples = grpo_save_state["consumed_samples"]
    val_period = master_config["grpo"]["val_period"]
    val_at_start = master_config["grpo"]["val_at_start"]
    colocated_inference = master_config["policy"]["generation"]["colocated"]["enabled"]

    # Run validation at the start if configured
    if val_at_start and step == 0:
        print("\n🔍 Running initial validation...")
        if NEED_REFIT and POLICY_GENERATION_STALE:
            refit_policy_generation(policy, policy_generation, colocated_inference)
            POLICY_GENERATION_STALE = False
        else:
            policy_generation.prepare_for_generation()
        val_metrics, validation_timings = validate(
            policy_generation,
            val_dataloader,
            tokenizer,
            val_task_to_env,
            step=0,
            master_config=master_config,
        )
        policy_generation.finish_generation()
        logger.log_metrics(val_metrics, step, prefix="validation")
        logger.log_metrics(validation_timings, step, prefix="timing/validation")

    # Run grpo training (single-turn)
    batch: BatchedDataDict[DatumSpec]
    for batch in dataloader:
        print(
            f"\n{'=' * 25} Step {step + 1}/{min(len(dataloader), master_config['grpo']['max_num_steps'])} {'=' * 25}"
        )
        maybe_gpu_profile_step(policy, step + 1)
        if policy != policy_generation:
            maybe_gpu_profile_step(policy_generation, step + 1)
        val_metrics, validation_timings = None, None

        with timer.time("total_step_time"):
            # Prepare batch
            print("▶ Preparing batch...")
            with timer.time("data_processing"):
                # Repeat batch items
                repeated_batch: BatchedDataDict[DatumSpec] = batch.repeat_interleave(
                    master_config["grpo"]["num_generations_per_prompt"]
                )
                # Convert LLMMessageLogType to FlatMessagesType for generation
                batched_flat, input_lengths = batched_message_log_to_flat_message(
                    repeated_batch["message_log"],
                    pad_value_dict={"token_ids": tokenizer.pad_token_id},
                )
                input_ids = batched_flat["token_ids"]

            # Generate responses - this updates the LLMMessageLogType in repeated_batch
            print(f"▶ Generating responses for batch of size {repeated_batch.size}...")
            with timer.time("prepare_for_generation"):
                if NEED_REFIT and POLICY_GENERATION_STALE:
                    # Compute KV scales if needed for FP8 quantization
                    if sync_kv_scales and kv_scales_cache is None:
                        print("[KV_SCALES] Computing KV cache scales for the first time...")
                        kv_scales_cache = compute_kv_scales_with_data(
                            policy, repeated_batch, master_config
                        )
                    
                    refit_policy_generation(
                        policy, policy_generation, colocated_inference, timer=timer,
                        kv_scales=kv_scales_cache if sync_kv_scales else None
                    )
                    POLICY_GENERATION_STALE = False
                else:
                    policy_generation.prepare_for_generation()

            with timer.time("generation"):
                # Use async rollouts if vLLM async engine is enabled
                if _should_use_async_rollouts(master_config):
                    (
                        repeated_batch,
                        rollout_metrics,
                    ) = run_async_multi_turn_rollout(
                        policy_generation=policy_generation,
                        input_batch=repeated_batch,
                        tokenizer=tokenizer,
                        task_to_env=task_to_env,
                        max_seq_len=master_config["policy"][
                            "max_total_sequence_length"
                        ],
                        max_rollout_turns=master_config["grpo"]["max_rollout_turns"],
                        greedy=False,
                    )
                else:
                    repeated_batch, rollout_metrics = run_multi_turn_rollout(
                        policy_generation=policy_generation,
                        input_batch=repeated_batch,
                        tokenizer=tokenizer,
                        task_to_env=task_to_env,
                        max_seq_len=master_config["policy"][
                            "max_total_sequence_length"
                        ],
                        max_rollout_turns=master_config["grpo"]["max_rollout_turns"],
                        greedy=False,
                    )
                policy_generation.finish_generation()

            # Calculate rewards & advantages
            print("▶ Processing rewards...")
            with timer.time("reward_calculation"):
                # Extract rewards from final_batch
                rewards = repeated_batch["total_reward"]

                print("▶ Computing advantages...")
                baseline, std = calculate_baseline_and_std_per_prompt(
                    input_ids,
                    rewards,
                    torch.ones_like(rewards),
                    leave_one_out_baseline=master_config["grpo"][
                        "use_leave_one_out_baseline"
                    ],
                )
                advantages = (rewards - baseline).unsqueeze(-1)

                if master_config["grpo"]["normalize_rewards"]:
                    # don't sharpen the ones with no variation
                    zero_std_mask = std > 0
                    advantages[zero_std_mask] = (
                        advantages[zero_std_mask] / std.unsqueeze(-1)[zero_std_mask]
                    )

            with timer.time("data_processing"):
                # Add loss mask and advantages to each message in LLMMessageLogType
                for i, message_log in enumerate(repeated_batch["message_log"]):
                    for j, message in enumerate(message_log):
                        if message["role"] == "assistant":
                            message["token_loss_mask"] = torch.ones_like(
                                message["token_ids"]
                            )
                        else:
                            message["token_loss_mask"] = torch.zeros_like(
                                message["token_ids"]
                            )
                        if "generation_logprobs" not in message:
                            message["generation_logprobs"] = torch.zeros_like(
                                message["token_ids"], dtype=torch.float32
                            )
                        message["advantages"] = advantages[i].expand(
                            message["token_ids"].shape
                        )

                # Convert updated LLMMessageLogType to FlatMessagesType for training
                flat_messages, input_lengths = batched_message_log_to_flat_message(
                    repeated_batch["message_log"],
                    pad_value_dict={"token_ids": tokenizer.pad_token_id},
                    make_sequence_length_divisible_by=master_config["policy"][
                        "make_sequence_length_divisible_by"
                    ],
                )

                # Create training data from flattened messages
                train_data = BatchedDataDict[ClippedPGLossDataDict](
                    {
                        "input_ids": flat_messages["token_ids"],
                        "input_lengths": input_lengths,
                        "advantages": flat_messages["advantages"],
                        "generation_logprobs": flat_messages["generation_logprobs"],
                        "token_mask": flat_messages["token_loss_mask"],
                        "sample_mask": repeated_batch["loss_multiplier"],
                    }
                )
                # this will be mini-batched inside the policy, so maintain the packed multimodal structure
                train_data.update(flat_messages.get_multimodal_dict(as_tensors=False))
                train_data.to("cpu")

            print("▶ Preparing for logprob inference...")
            with timer.time("logprob_inference_prep"):
                policy.prepare_for_lp_inference()

            print("▶ Computing logprobs...")
            with timer.time("policy_and_reference_logprobs"):
                fprop_logprobs = policy.get_logprobs(train_data)["logprobs"]
                reference_logprobs = policy.get_reference_policy_logprobs(train_data)[
                    "reference_logprobs"
                ]
                train_data["prev_logprobs"] = fprop_logprobs
                train_data["reference_policy_logprobs"] = reference_logprobs

            print("▶ Preparing for training...")
            with timer.time("training_prep"):
                policy.prepare_for_training()  # set model train and reload optim to GPU
                POLICY_GENERATION_STALE = True

            print("▶ Training policy...")
            with timer.time("policy_training"):
                train_results = policy.train(train_data, loss_fn)

            # Recompute KV scales after policy training if needed
            if sync_kv_scales:
                print("[KV_SCALES] Recomputing KV cache scales after policy update...")
                kv_scales_cache = compute_kv_scales_with_data(
                    policy, repeated_batch, master_config
                )
                # Set generation as stale to force refit with new scales
                POLICY_GENERATION_STALE = True

            is_last_step = step + 1 == min(
                master_config["grpo"]["max_num_steps"], len(dataloader)
            )

            # Run validation if it's a validation step
            if val_period > 0 and (step + 1) % val_period == 0:
                if NEED_REFIT and POLICY_GENERATION_STALE:
                    refit_policy_generation(
                        policy, policy_generation, colocated_inference
                    )
                    POLICY_GENERATION_STALE = False
                else:
                    policy_generation.prepare_for_generation()
                val_metrics, validation_timings = validate(
                    policy_generation,
                    val_dataloader,
                    tokenizer,
                    val_task_to_env,
                    step=step + 1,
                    master_config=master_config,
                )
                policy_generation.finish_generation()
                logger.log_metrics(
                    validation_timings, step + 1, prefix="timing/validation"
                )
                logger.log_metrics(val_metrics, step + 1, prefix="validation")

            ## Checkpointing
            consumed_samples += master_config["grpo"]["num_prompts_per_step"]
            timeout.mark_iteration()

            should_save_by_step = (
                is_last_step
                or (step + 1) % master_config["checkpointing"]["save_period"] == 0
            )
            # +1 because step is 0-indexed
            # Check if timeout-based checkpointing is enabled in config.
            should_save_by_timeout = timeout.check_save()

            if master_config["checkpointing"]["enabled"] and (
                should_save_by_step or should_save_by_timeout
            ):
                policy.prepare_for_training()

                grpo_save_state["step"] = step + 1
                if val_metrics is not None:
                    grpo_save_state["val_reward"] = val_metrics["accuracy"]
                elif "val_reward" in grpo_save_state:
                    del grpo_save_state["val_reward"]
                grpo_save_state["consumed_samples"] = consumed_samples

                if master_config["checkpointing"]["metric_name"] is not None:
                    if (
                        master_config["checkpointing"]["metric_name"]
                        not in grpo_save_state
                    ):
                        warnings.warn(
                            f"You asked to save checkpoints based on {master_config['checkpointing']['metric_name']} but the metric is not found in the save state. "
                            "Saving most recent k checkpoints instead."
                        )
                        master_config["checkpointing"]["metric_name"] = None

                with timer.time("checkpointing"):
                    print(f"Saving checkpoint for step {step + 1}...")
                    checkpoint_path = checkpointer.init_tmp_checkpoint(
                        step + 1, grpo_save_state, master_config
                    )
                    policy.save_checkpoint(
                        weights_path=os.path.join(checkpoint_path, "policy", "weights"),
                        optimizer_path=os.path.join(
                            checkpoint_path, "policy", "optimizer"
                        ),
                        tokenizer_path=os.path.join(
                            checkpoint_path, "policy", "tokenizer"
                        ),
                    )
                    torch.save(
                        dataloader.state_dict(),
                        os.path.join(checkpoint_path, "train_dataloader.pt"),
                    )
                    checkpointer.finalize_checkpoint(checkpoint_path)

        # Logging
        # Log training data
        log_data = {"content": flat_messages["content"]}
        log_data["rewards"] = rewards.tolist()
        log_data["generation_logprobs"] = train_data["generation_logprobs"].tolist()
        log_data["prev_logprobs"] = train_data["prev_logprobs"].tolist()
        log_data["input_lengths"] = input_lengths.tolist()
        logger.log_batched_dict_as_jsonl(log_data, f"train_data_step{step}.jsonl")

        metrics = {
            "loss": train_results["loss"].numpy(),
            "reward": rewards.numpy(),
            "grad_norm": train_results["grad_norm"].numpy(),
            "mean_prompt_length": repeated_batch["length"].numpy(),
            "total_num_tokens": input_lengths.numpy(),
        }
        metrics.update(train_results["all_mb_metrics"])
        for k, v in metrics.items():
            if k in {
                "lr",
                "wd",
                "reward",
                "global_valid_seqs",
                "global_valid_toks",
                "mean_prompt_length",
            }:
                metrics[k] = np.mean(v).item()
            else:
                metrics[k] = np.sum(v).item()
        metrics.update(rollout_metrics)

        timing_metrics: dict[str, float] = timer.get_timing_metrics(reduction_op="sum")  # type: ignore
        # track example with high token mult prob error above 1.05
        if metrics["token_mult_prob_error"] > 1.05:
            logger.log_plot_token_mult_prob_error(
                {
                    "prompt_lengths": repeated_batch["length"],
                    "full_lengths": input_lengths,
                    "generation_logprobs": train_data["generation_logprobs"],
                    "prev_logprobs": train_data["prev_logprobs"],
                    "token_mask": train_data["token_mask"],
                    "sample_mask": train_data["sample_mask"],
                },
                step + 1,
                name="train/token_mult_prob_error_plot_sample",
            )

        print("\n📊 Training Results:")

        print(f"  • Loss: {metrics['loss']:.4f}")
        print(f"  • Avg Reward: {np.mean(rewards.numpy()):.4f}")
        print(
            f"  • Mean Generation Length: {rollout_metrics['mean_gen_tokens_per_sample']:.4f}"
        )
        if "total_flops" in train_results:
            total_tflops = (
                train_results["total_flops"] / timing_metrics["policy_training"] / 1e12
            )
            num_ranks = train_results["num_ranks"]
            print(
                f"  • Training FLOPS: {total_tflops:.2f} TFLOPS ({total_tflops / num_ranks:.2f} TFLOPS per rank)"
            )
            if "theoretical_tflops" in train_results:
                theoretical_tflops = train_results["theoretical_tflops"]
                print(
                    f"  • Training Model Floating Point Utilization: {100 * total_tflops / theoretical_tflops:.2f}%"
                )
                metrics["train_fp_utilization"] = total_tflops / theoretical_tflops

        print("\n⏱️  Timing:")
        # Display total time first, separately
        total_time = timing_metrics.get("total_step_time", 0)

        total_num_gpus = (
            master_config["cluster"]["num_nodes"]
            * master_config["cluster"]["gpus_per_node"]
        )
        metrics.update(
            {
                "tokens_per_sec_per_gpu": metrics["total_num_tokens"]
                / total_time
                / total_num_gpus
            }
        )

        print(f"  • Total step time: {total_time:.2f}s")

        # Display all other timing metrics
        for k, v in sorted(
            timing_metrics.items(), key=lambda item: item[1], reverse=True
        ):
            if k != "total_step_time":
                percent = (v / total_time * 100) if total_time > 0 else 0
                print(f"  • {k}: {v:.2f}s ({percent:.1f}%)")

        logger.log_metrics(metrics, step + 1, prefix="train")
        logger.log_metrics(timing_metrics, step + 1, prefix="timing/train")

        timer.reset()
        step += 1
        if step >= master_config["grpo"]["max_num_steps"]:
            break


def validate(
    policy_generation: GenerationInterface,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer,
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    step: int,
    master_config: MasterConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run validation on the validation dataset."""
    if val_dataloader is None:
        print("  ⚠️ No validation dataloader provided, skipping validation")
        return {}, {}

    timer = Timer()
    with timer.time("total_validation_time"):
        print(f"▶ Starting validation at step {step}...")

        total_rewards = []
        total_lengths = []
        all_message_logs = []  # Collect all message logs

        max_batches = (
            master_config["grpo"]["max_val_samples"]
            // master_config["grpo"]["val_batch_size"]
        )
        for batch_idx, val_batch in enumerate(val_dataloader):
            if batch_idx >= max_batches:
                break

            # Generate responses (updates the LLMMessageLogType in batch_with_msg_logs)
            # Use async rollouts if vLLM async engine is enabled
            if _should_use_async_rollouts(master_config):
                val_batch, gen_metrics = run_async_multi_turn_rollout(
                    policy_generation,
                    val_batch,
                    tokenizer,
                    val_task_to_env,
                    max_seq_len=master_config["policy"]["max_total_sequence_length"],
                    max_rollout_turns=master_config["grpo"]["max_rollout_turns"],
                    greedy=False,
                )
            else:
                val_batch, gen_metrics = run_multi_turn_rollout(
                    policy_generation,
                    val_batch,
                    tokenizer,
                    val_task_to_env,
                    max_seq_len=master_config["policy"]["max_total_sequence_length"],
                    max_rollout_turns=master_config["grpo"]["max_rollout_turns"],
                    greedy=False,
                )
            rewards = val_batch["total_reward"]

            total_rewards.extend(rewards.tolist())
            total_lengths.append(gen_metrics["mean_gen_tokens_per_sample"])

            # Collect message logs for later display
            to_env = [
                get_keys_from_message_log(
                    val_batch["message_log"][i], ["role", "content"]
                )
                for i in range(len(val_batch["message_log"]))
            ]

            all_message_logs.extend(to_env)

        # Calculate validation metrics
        accuracy = sum(total_rewards) / len(total_rewards)
        avg_length = sum(total_lengths) / len(total_lengths)

        val_metrics = {
            "accuracy": accuracy,
            "avg_length": avg_length,
        }

        # Print sample conversations only once at the end of validation
        try:
            print_message_log_samples(
                all_message_logs,
                total_rewards,
                num_samples=min(
                    master_config["logger"]["num_val_samples_to_print"],
                    len(all_message_logs),
                ),
                step=step,
            )
        except Exception as e:
            print(f"\n  ⚠️ Error displaying message samples: {str(e)}")
            print("  ⚠️ Continuing validation without displaying samples...")

    # Get timing metrics
    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    validation_time = timing_metrics.get("total_validation_time", 0)

    # Print summary of validation results
    print("\n📊 Validation Results:")
    print(f"    • Accuracy: {accuracy:.4f}")
    print(f"    • Average response length: {avg_length:.1f} tokens")
    print(f"    • Samples processed: {len(total_rewards)}")

    # Print timing information
    print("\n  ⏱️  Validation Timing:")
    validation_time = timing_metrics.get("total_validation_time", 0)
    print(f"    • Total validation time: {validation_time:.2f}s")

    # Make sure to reset the timer after validation
    timer.reset()

    return val_metrics, timing_metrics
