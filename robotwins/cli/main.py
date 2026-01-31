import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="robotwins")
    parser.add_argument(
        "command",
        choices=["init", "calibrate", "train", "eval", "viewer"],
        help="Command to run.",
    )
    parser.add_argument("--config", default="", help="Path to a run config.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    print(f"Command '{args.command}' is not implemented yet. See docs/PRD.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
