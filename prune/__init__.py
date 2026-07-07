"""Expert pruning of the MoE actor (see prune/README.md).

Pipeline: collect_stats -> select -> apply -> validate. Pure logic lives in
plan.py / remap.py (no torch) so it is unit-testable on CPU-only machines.
"""
