"""Train or recover RoboPolicy using the public library interface."""

from __future__ import annotations

import argparse

from robopolicy import Trainer, load_config


def main() -> None:
    """Parse a configuration or recovery checkpoint and call the unified trainer."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument("--resume")
    parser.add_argument("--dataset-root")
    parser.add_argument("--device")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.resume:
        if args.dry_run:
            parser.error("--dry-run requires --config")
        trainer = Trainer.resume(args.resume, dataset_root=args.dataset_root, device=args.device)
    else:
        if args.dataset_root or args.device:
            parser.error("Use configuration settings for fresh training")
        config = load_config(args.config)
        if args.dry_run:
            print(config.model_dump_json(indent=2))
            return
        trainer = Trainer(config)
    trainer.fit()


if __name__ == "__main__":
    main()
