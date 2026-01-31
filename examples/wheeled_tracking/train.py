import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robotwins.rl.trainers.minimal import run_training


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="examples/wheeled_tracking/config.yaml",
        help="Path to run config.",
    )
    parser.add_argument("--steps", type=int, default=50, help="Training steps.")
    args = parser.parse_args()
    total_reward, steps = run_training(args.config, steps=args.steps)
    print(f"training finished: steps={steps} total_reward={total_reward:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
