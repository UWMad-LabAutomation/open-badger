"""Download/open a LeRobot dataset and ensure state/action quantile statistics."""

from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from open_badger.data.preparation import ensure_quantile_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("huggingface", "local"),
        default="huggingface",
        help="Download from Hugging Face or prepare a dataset already on disk.",
    )
    parser.add_argument("--dataset-id", required=True, help="LeRobot dataset ID.")
    parser.add_argument("--data-root", type=Path, help="Dataset directory or download destination.")
    parser.add_argument("--revision", help="Hugging Face dataset revision.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
        help="Number of raw rows per batch while computing statistics.",
    )
    args = parser.parse_args()

    if args.backend == "local" and args.data_root is None:
        parser.error("--data-root is required with --backend local.")
    if (
        args.backend == "local"
        and args.data_root is not None
        and not (args.data_root / "meta" / "info.json").is_file()
    ):
        parser.error(f"LeRobot metadata not found under local data root: {args.data_root}")

    dataset = LeRobotDataset(
        repo_id=args.dataset_id,
        root=args.data_root,
        revision=args.revision,
    )
    updated = ensure_quantile_stats(dataset, batch_size=args.batch_size)
    if updated:
        print(
            f"Computed and saved quantile statistics for {', '.join(updated)} "
            f"under {dataset.meta.root}."
        )
    else:
        print(f"Quantile statistics already exist under {dataset.meta.root}.")


if __name__ == "__main__":
    main()
