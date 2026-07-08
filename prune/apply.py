"""Stage 3: apply a prune plan to a checkpoint's safetensors.

Purely mechanical: drops pruned experts' tensors, renumbers the kept ones to
contiguous indices, slices the router gate rows, and updates ``num_experts``
in config.json (flat actor configs and the vLLM-wrapped ``text_config`` both
handled). Works on the bf16 actor and on compressed-tensors checkpoints alike
— the W8A8 candidate's per-expert quantization tensors (weight_scale, ...) are
renamed together with their module, and the router gate is bf16 (quantization
ignore list) so slicing it is safe.

No forward passes, no GPU: tensors are streamed shard by shard on CPU.

  python -m prune.apply \
    --model checkpoints/quantized/qwen3_5_35b_a3b_w8a8 \
    --plan checkpoints/pruned/qwen3_5_35b_a3b_w8a8/prune_plan.json \
    --output checkpoints/pruned/qwen3_5_35b_a3b_w8a8/model
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from prune.remap import PlanIndex, plan_tensor, update_config_num_experts

_INDEX_NAME = "model.safetensors.index.json"
_SINGLE_NAME = "model.safetensors"
# Never copied into the pruned checkpoint verbatim.
_SKIP_COPY = {"config.json", _INDEX_NAME}


def _shards(model_dir: Path) -> list[Path]:
    index = model_dir / _INDEX_NAME
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        return [model_dir / f for f in sorted(set(weight_map.values()))]
    single = model_dir / _SINGLE_NAME
    if single.exists():
        return [single]
    raise FileNotFoundError(f"no {_INDEX_NAME} or {_SINGLE_NAME} in {model_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Checkpoint dir to prune (the candidate).")
    ap.add_argument("--plan", required=True, help="prune_plan.json from prune.select.")
    ap.add_argument("--output", required=True, help="Output checkpoint dir.")
    args = ap.parse_args()

    from safetensors.torch import load_file, save_file

    model_dir = Path(args.model)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = json.loads(Path(args.plan).read_text())
    index = PlanIndex(plan)

    # --- rewrite shards -----------------------------------------------------
    weight_map: dict[str, str] = {}
    total_size = 0
    dropped = renamed = sliced_gate = sliced_expert = kept = 0
    shard_paths = _shards(model_dir)
    for shard_path in shard_paths:
        tensors = load_file(str(shard_path))
        out_tensors = {}
        for name, t in tensors.items():
            kind, arg = plan_tensor(name, index)
            if kind == "drop":
                dropped += 1
                continue
            if kind == "rename":
                out_tensors[arg] = t
                renamed += 1
            elif kind == "slice0":
                if t.shape[0] != index.num_experts_orig:
                    raise ValueError(
                        f"{name}: dim0={t.shape[0]} != num_experts="
                        f"{index.num_experts_orig}; refusing to slice"
                    )
                out_tensors[name] = t[arg].contiguous()
                if ".mlp.gate." in name:
                    sliced_gate += 1
                else:
                    sliced_expert += 1
            else:
                out_tensors[name] = t
                kept += 1
        for name, t in out_tensors.items():
            weight_map[name] = shard_path.name
            total_size += t.numel() * t.element_size()
        save_file(out_tensors, str(out_dir / shard_path.name), metadata={"format": "pt"})
        print(f"[apply] {shard_path.name}: wrote {len(out_tensors)} tensors")

    print(
        f"[apply] tensors: kept={kept} renamed={renamed} "
        f"sliced(gate)={sliced_gate} sliced(expert)={sliced_expert} "
        f"dropped={dropped}"
    )
    experts_touched = renamed > 0 or dropped > 0 or sliced_expert > 0
    if not experts_touched or sliced_gate == 0:
        raise RuntimeError(
            "no expert/gate tensors were touched — tensor names did not match "
            "the expected layout (see prune/remap.py)"
        )

    if len(shard_paths) > 1 or (model_dir / _INDEX_NAME).exists():
        (out_dir / _INDEX_NAME).write_text(
            json.dumps(
                {"metadata": {"total_size": total_size}, "weight_map": weight_map},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    # --- config + aux files ---------------------------------------------------
    cfg = json.loads((model_dir / "config.json").read_text())
    changed = update_config_num_experts(cfg, index.keep_per_layer)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"[apply] config: {', '.join(changed)} -> {index.keep_per_layer}")

    for f in model_dir.iterdir():
        if f.is_file() and f.suffix != ".safetensors" and f.name not in _SKIP_COPY:
            shutil.copy2(f, out_dir / f.name)

    (out_dir / "expert_pruning.json").write_text(
        json.dumps(
            {
                "source_model": str(model_dir),
                "plan": str(Path(args.plan)),
                "num_experts_orig": index.num_experts_orig,
                "keep_per_layer": index.keep_per_layer,
                "report": plan.get("report", {}),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[apply] pruned checkpoint -> {out_dir}")


if __name__ == "__main__":
    main()
