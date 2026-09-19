"""Structural equivalence check for the models/ package move (T1).

Usage: python .scratch/pluggable-encoders/equiv_check.py old|new

Run each mode in its own process so Keras' global layer-naming counter starts
from the same state; identical output means the move changed no structure.
"""

import subprocess
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import config as C

BUILD_KWARGS = dict(
    categorical_cols=C.CATEGORICAL_COLS,
    numeric_cols=C.NUMERIC_COLS,
    cat_vocab_size=2385,
    num_tasks=4,
)


def old_builder():
    """The builder as it existed before the move, read from git history."""
    source = subprocess.run(
        ["git", "show", "HEAD:MMoE_model.py"],
        capture_output=True,
        encoding="utf-8",
    ).stdout
    module = types.ModuleType("old_mmoe")
    exec(compile(source, "old_mmoe", "exec"), module.__dict__)
    return module.build_mmoe_model


def new_builder():
    """The builder as it exists in the models package."""
    from models import build_mmoe_model

    return build_mmoe_model


def main():
    which = sys.argv[1]
    builder = {"old": old_builder, "new": new_builder}[which]
    model = builder()(**BUILD_KWARGS)
    print("inputs:", "|".join(inp.name for inp in model.inputs))
    print("outputs:", "|".join(out.name for out in model.outputs))
    print("layers:", "|".join(layer.name for layer in model.layers))
    print("params:", model.count_params())
    print("weights:", "|".join(str(tuple(w.shape)) for w in model.get_weights()))


if __name__ == "__main__":
    main()
