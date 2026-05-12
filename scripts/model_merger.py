"""HF ↔ Megatron weight bridge; must run under torchrun (see model_merger.sh).

Writes:
  * HuggingFace-format weights via mbridge (rank 0 only), for vLLM / HF tooling.
  * Megatron-Core *distributed* checkpoint via dist_checkpointing.save (all ranks),
    compatible with verl's load_mcore_dist_weights when actor.megatron.use_dist_checkpointing
    is true — point dist_checkpointing_path at the directory passed as --megatron-dist-dir.
"""

from __future__ import annotations

import argparse
import os

import megatron.core
import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core import tensor_parallel as mtp
from mbridge import AutoBridge
from packaging import version

from verl.utils.megatron.dist_checkpointing import save_dist_checkpointing
from verl.utils.megatron_utils import unwrap_model


def _init_distributed() -> None:
    if dist.is_initialized():
        return
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise SystemExit(
            "Distributed env not set. Launch via:\n"
            "  bash scripts/model_merger.sh\n"
            "or:\n"
            "  torchrun --standalone --nproc_per_node=8 scripts/model_merger.py"
        )
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")


def _dist_ckpt_content_metadata() -> dict:
    """Match verl MegatronCheckpointManager._build_sharded_state_dict_metadata (model shard only)."""
    mcore_ge_014 = version.parse(megatron.core.__version__) >= version.parse("0.14.0")
    if not mcore_ge_014:
        return {"distrib_optim_sharding_type": "fully_sharded_model_space"}
    return {
        "singleton_local_shards": False,
        "chained_optim_avoid_prefix": True,
    }


def save_megatron_distributed_checkpoint(models: list, dist_dir: str) -> None:
    """Collective Megatron-Core dist checkpoint (weights only, ``model`` key)."""
    if dist_dir:
        os.makedirs(dist_dir, exist_ok=True)
    torch.distributed.barrier()

    content_metadata = _dist_ckpt_content_metadata()
    state_dict: dict = {}
    unwrapped = unwrap_model(models)
    if not isinstance(unwrapped, list):
        unwrapped = [unwrapped]

    for vpp_rank, model in enumerate(unwrapped):
        key = f"model{vpp_rank}" if len(unwrapped) > 1 else "model"
        model_metadata = dict(content_metadata)
        model_metadata["dp_cp_group"] = mpu.get_data_parallel_group(with_context_parallel=True)
        state_dict[key] = model.sharded_state_dict(metadata=model_metadata)

    save_dist_checkpointing(
        sharded_state_dict=state_dict,
        ckpt_path=dist_dir,
        async_save=False,
        content_metadata=content_metadata,
    )
    torch.distributed.barrier()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-model",
        default=os.environ.get("HF_MODEL_PATH", "Qwen/Qwen3.5-35B-A3B"),
        help="HF hub id or local path to load from (mbridge).",
    )
    parser.add_argument(
        "--hf-out",
        default=os.environ.get("HF_OUT", "checkpoints/qwen3_5_35b_TP2_EP8_PP1"),
        help="Directory for HF safetensors + config (rank 0 only).",
    )
    parser.add_argument(
        "--megatron-dist-dir",
        default=os.environ.get("MEGATRON_DIST_DIR", ""),
        help=(
            "If non-empty, all ranks also save a Megatron dist_checkpointing tree here "
            "(use this path as actor.megatron.dist_checkpointing_path in verl). "
            "Default: <hf-out>/dist_ckpt when MEGATRON_DIST_DIR is unset and --save-megatron-dist is set."
        ),
    )
    parser.add_argument(
        "--save-hf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export HF weights via mbridge.save_weights (default: true).",
    )
    parser.add_argument(
        "--save-megatron-dist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export Megatron-Core distributed checkpoint (default: true).",
    )
    args = parser.parse_args()

    _init_distributed()

    mpu.initialize_model_parallel(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=1,
    )

    mtp.model_parallel_cuda_manual_seed(
        int(os.environ.get("SEED", "42")),
        force_reset_rng=True,
    )

    bridge = AutoBridge.from_pretrained(args.hf_model)
    models = bridge.get_model(weight_path=args.hf_model)

    dist_dir = args.megatron_dist_dir.strip()
    if args.save_megatron_dist and not dist_dir:
        dist_dir = os.path.join(args.hf_out, "dist_ckpt")

    # Megatron dist_checkpointing.save is collective — run before rank-0-only HF export.
    if args.save_megatron_dist:
        save_megatron_distributed_checkpoint(models, dist_dir)
        if dist.is_initialized() and dist.get_rank() == 0:
            print(f"Saved Megatron distributed checkpoint under {os.path.abspath(dist_dir)}")

    if args.save_hf:
        bridge.save_weights(
            models,
            args.hf_out,
            memory_efficient=False,
        )
        torch.distributed.barrier()
        if dist.is_initialized() and dist.get_rank() == 0:
            print(f"Saved HF checkpoint under {os.path.abspath(args.hf_out)}")


if __name__ == "__main__":
    main()
