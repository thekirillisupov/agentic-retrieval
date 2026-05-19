"""Export HuggingFace weights from a verl Megatron-Core *distributed* actor checkpoint.

Expects ``<actor_dir>/dist_ckpt`` (from training with ``use_dist_checkpointing``) and
``<actor_dir>/huggingface`` (config + tokenizer from the run). Uses the same TP/EP/PP
layout as training (defaults match ``configs/gspo_qwen3_moe.yaml``).

Run under torchrun with one process per data-parallel rank (default 8 GPUs for TP=2, EP=8):

  bash scripts/dist_ckpt_to_hf.sh --actor-dir checkpoints/.../global_step_60/actor
"""

from __future__ import annotations

import argparse
import os

import megatron.core
import torch
import torch.distributed as dist
from megatron.core import dist_checkpointing as mcore_dist_checkpointing
from megatron.core import parallel_state as mpu
from megatron.core import tensor_parallel as mtp
from mbridge import AutoBridge
from packaging import version

from mbridge.core.safetensor_io import SafeTensorIO
from verl.utils.megatron.dist_checkpointing import load_dist_checkpointing
from verl.utils.megatron_utils import unwrap_model


def _init_distributed() -> None:
    if dist.is_initialized():
        return
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise SystemExit(
            "Distributed env not set. Launch via:\n"
            "  bash scripts/dist_ckpt_to_hf.sh\n"
            "or:\n"
            "  torchrun --standalone --nproc_per_node=8 scripts/dist_ckpt_to_hf.py ..."
        )
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")


def _dist_ckpt_content_metadata_fallback() -> dict:
    """Match ``scripts/model_merger.py`` / verl when checkpoint has no content metadata."""
    mcore_ge_014 = version.parse(megatron.core.__version__) >= version.parse("0.14.0")
    if not mcore_ge_014:
        return {"distrib_optim_sharding_type": "fully_sharded_model_space"}
    return {
        "singleton_local_shards": False,
        "chained_optim_avoid_prefix": True,
    }


def _resolve_sharded_sd_metadata(dist_dir: str) -> dict:
    load_content_metadata = getattr(mcore_dist_checkpointing, "load_content_metadata", None)
    if load_content_metadata is None:
        return _dist_ckpt_content_metadata_fallback()
    meta = load_content_metadata(checkpoint_dir=dist_dir)
    if meta is None:
        return _dist_ckpt_content_metadata_fallback()
    return meta


def _build_model_only_sharded_state_dict(models: list, dist_dir: str) -> dict:
    """Sharded state dict for *model* tensors only (skips optimizer in the checkpoint)."""
    sharded_sd_metadata = _resolve_sharded_sd_metadata(dist_dir)
    state_dict: dict = {}
    unwrapped = unwrap_model(models)
    if not isinstance(unwrapped, list):
        unwrapped = [unwrapped]

    for vpp_rank, model in enumerate(unwrapped):
        key = f"model{vpp_rank}" if len(unwrapped) > 1 else "model"
        if hasattr(model, "module"):
            model = model.module
        model_metadata = dict(sharded_sd_metadata)
        model_metadata["dp_cp_group"] = mpu.get_data_parallel_group(with_context_parallel=True)
        state_dict[key] = model.sharded_state_dict(metadata=model_metadata)
    return state_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_actor = os.environ.get(
        "ACTOR_DIR",
        "checkpoints/gspo_qwen3_moe/global_step_60/actor",
    )
    parser.add_argument(
        "--actor-dir",
        default=default_actor,
        help="Directory containing dist_ckpt/ and huggingface/ (verl actor checkpoint).",
    )
    parser.add_argument(
        "--dist-dir",
        default="",
        help="Override path to Megatron dist checkpoint (default: <actor-dir>/dist_ckpt).",
    )
    parser.add_argument(
        "--hf-template",
        default=os.environ.get("HF_TEMPLATE", ""),
        help="HF path for mbridge (config/architecture). Default: <actor-dir>/huggingface.",
    )
    parser.add_argument(
        "--hf-index",
        default=os.environ.get("HF_INDEX", ""),
        help=(
            "Path to an HF model that has model.safetensors.index.json, used to "
            "determine output shard layout (memory-efficient save). "
            "Default: same as --hf-template. Pass the base model snapshot dir "
            "when the actor huggingface/ only contains config+tokenizer."
        ),
    )
    parser.add_argument(
        "--hf-out",
        default=os.environ.get("HF_OUT", ""),
        help="Output directory for HF safetensors + config. Default: <parent of actor-dir>/hf_model.",
    )
    parser.add_argument(
        "--tensor-model-parallel-size",
        type=int,
        default=int(os.environ.get("TP", "2")),
    )
    parser.add_argument(
        "--pipeline-model-parallel-size",
        type=int,
        default=int(os.environ.get("PP", "1")),
    )
    parser.add_argument(
        "--expert-model-parallel-size",
        type=int,
        default=int(os.environ.get("EP", "8")),
    )
    parser.add_argument(
        "--expert-tensor-parallel-size",
        type=int,
        default=int(os.environ.get("ETP", "1")),
    )
    args = parser.parse_args()

    actor_dir = os.path.abspath(args.actor_dir)
    dist_dir = args.dist_dir.strip() or os.path.join(actor_dir, "dist_ckpt")
    dist_dir = os.path.abspath(dist_dir)
    hf_template = args.hf_template.strip() or os.path.join(actor_dir, "huggingface")
    hf_template = os.path.abspath(hf_template)
    hf_index = args.hf_index.strip() or hf_template
    hf_index = os.path.abspath(hf_index)

    hf_out = args.hf_out.strip()
    if not hf_out:
        hf_out = os.path.join(os.path.dirname(actor_dir), "hf_model")
    hf_out = os.path.abspath(hf_out)

    meta_path = os.path.join(dist_dir, ".metadata")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing Megatron dist checkpoint metadata: {meta_path}")

    _init_distributed()

    mpu.initialize_model_parallel(
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        expert_model_parallel_size=args.expert_model_parallel_size,
        expert_tensor_parallel_size=args.expert_tensor_parallel_size,
    )

    mtp.model_parallel_cuda_manual_seed(
        int(os.environ.get("SEED", "42")),
        force_reset_rng=True,
    )

    bridge = AutoBridge.from_pretrained(hf_template)
    # When hf_index has a full model (safetensors + index), load it as the initial
    # weight state.  This pre-fills weights that verl does NOT fine-tune (mtp.*,
    # visual encoder if frozen, etc.) so the output model is complete.
    # bridge.safetensor_io is set correctly by get_model, giving save_weights the
    # right sharding index.  When hf_index == hf_template (no full model available),
    # fall back to random init + manual SafeTensorIO.
    hf_index_has_weights = os.path.isfile(os.path.join(hf_index, "model.safetensors.index.json"))
    if hf_index_has_weights:
        models = bridge.get_model(weight_path=hf_index)
    else:
        models = bridge.get_model()
        bridge.safetensor_io = SafeTensorIO(hf_index)

    sharded_state_dict = _build_model_only_sharded_state_dict(models, dist_dir)
    torch.distributed.barrier()
    loaded = load_dist_checkpointing(sharded_state_dict, dist_dir)
    torch.distributed.barrier()

    unwrapped = unwrap_model(models)
    if not isinstance(unwrapped, list):
        unwrapped = [unwrapped]
    for vpp_rank, model in enumerate(unwrapped):
        if len(unwrapped) == 1:
            chunk = loaded["model"]
        else:
            chunk = loaded[f"model{vpp_rank}"]
        mpu.set_virtual_pipeline_model_parallel_rank(vpp_rank)
        model.load_state_dict(chunk, strict=True)

    torch.distributed.barrier()
    # distributed_filesystem=True uses _save_weights_fast: each rank saves its own
    # TP/EP shards in parallel, then all ranks merge into final HF shard files.
    # This avoids rank 0 gathering the full model (~70 GB) into a single GPU.
    bridge.save_weights(models, hf_out, memory_efficient=True, distributed_filesystem=True)
    torch.distributed.barrier()
    if dist.is_initialized() and dist.get_rank() == 0:
        print(f"Wrote HuggingFace model for vLLM under {hf_out}")


if __name__ == "__main__":
    main()
