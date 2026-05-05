import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Export GAP_SAC actor policy.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="actor_policy.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from turtlebot3_drl.agents.networks import torch
    except ImportError as exc:
        print(f"Missing dependency for export: {exc}")
        return 2
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    torch.save({"actor": checkpoint["actor"]}, args.output)
    print(f"exported actor weights to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
