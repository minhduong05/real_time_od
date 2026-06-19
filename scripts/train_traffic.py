"""Run Intersection-Flow-5K training experiments."""

from __future__ import annotations

from _train_common import build_parser, train_from_config


def main() -> None:
    parser = build_parser(__doc__)
    parser.set_defaults(config="configs/experiments/traffic_yolov8n_p2.yaml")
    args = parser.parse_args()
    train_from_config(args)


if __name__ == "__main__":
    main()
