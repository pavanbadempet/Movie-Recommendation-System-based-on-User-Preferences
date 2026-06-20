"""Compatibility entry point for the serving-compatible RL trainer.

The former 771x768 synthetic trainer could overwrite the 20x16 checkpoint
consumed by serving. All training now delegates to train_rl_policy_compact.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_rl_policy_compact import ACTION_DIM as _ACTION_DIM
from scripts.train_rl_policy_compact import STATE_DIM as _STATE_DIM
from scripts.train_rl_policy_compact import _parse_args, train

ACTION_DIM = _ACTION_DIM
STATE_DIM = _STATE_DIM


def train_offline_rl(epochs: int = 200, lr: float = 1e-4, batch_size: int = 256) -> None:
    """Train the only supported serving-compatible policy."""
    train(epochs=epochs, lr=lr, batch_size=batch_size)


if __name__ == "__main__":
    args = _parse_args()
    train_offline_rl(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
