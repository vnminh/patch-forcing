#!/usr/bin/env python3
"""Merge resumed TensorBoard runs and plot VTON loss/attention trends."""

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def smooth(values, window):
    if len(values) < 2 or window <= 1:
        return values
    width = min(int(window), len(values))
    kernel = np.ones(width, dtype=np.float64) / width
    padded = np.pad(values, (width - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def load_scalars(log_root):
    merged = defaultdict(dict)
    event_files = sorted(
        glob.glob(os.path.join(log_root, "**", "events.out.tfevents*"), recursive=True),
        key=os.path.getmtime,
    )
    for event_file in event_files:
        accumulator = EventAccumulator(event_file, size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars(tag):
                previous = merged[tag].get(event.step)
                if previous is None or event.wall_time >= previous[0]:
                    merged[tag][event.step] = (event.wall_time, event.value)
    return {
        tag: (
            np.asarray(sorted(points), dtype=np.int64),
            np.asarray([points[step][1] for step in sorted(points)], dtype=np.float64),
        )
        for tag, points in merged.items()
    }, event_files


def add_series(ax, scalars, tags, window, labels=None):
    plotted = False
    for tag in tags:
        if tag not in scalars:
            continue
        steps, values = scalars[tag]
        label = labels.get(tag, tag.removeprefix("train/")) if labels else tag.removeprefix("train/")
        ax.plot(steps, smooth(values, window), label=label, linewidth=1.5)
        plotted = True
    if plotted:
        ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)
    ax.set_xlabel("optimizer step")


def aggregate_gradients(scalars):
    groups = defaultdict(list)
    for tag, (steps, values) in scalars.items():
        if not tag.startswith("train/garment_grad/block_"):
            continue
        scale = tag.split("/")[2].split("_")[-1]
        groups[scale].append((steps, values))
    output = {}
    for scale, series in groups.items():
        common = sorted(set.intersection(*(set(steps.tolist()) for steps, _ in series)))
        if not common:
            continue
        lookup = [{int(step): value for step, value in zip(steps, values)} for steps, values in series]
        output[f"gradient/{scale}"] = (
            np.asarray(common),
            np.asarray([np.mean([points[step] for points in lookup]) for step in common]),
        )
    return output


def trend_summary(scalars, recent_points):
    summary = {}
    for tag, (steps, values) in sorted(scalars.items()):
        if not any(term in tag for term in ("loss", "correspondence", "target_mass", "entropy", "coverage")):
            continue
        recent = values[-recent_points:]
        prior = values[-2 * recent_points : -recent_points]
        recent_steps = steps[-recent_points:]
        slope = 0.0
        if len(recent) > 1 and recent_steps[-1] != recent_steps[0]:
            slope = float(np.polyfit(recent_steps.astype(float), recent, 1)[0] * 1000)
        summary[tag] = {
            "last_step": int(steps[-1]),
            "last": float(values[-1]),
            "recent_mean": float(np.mean(recent)),
            "prior_mean": None if len(prior) == 0 else float(np.mean(prior)),
            "slope_per_1000_steps": slope,
        }
    return summary


def main(args):
    scalars, event_files = load_scalars(args.log_root)
    if not scalars:
        raise ValueError(f"No TensorBoard scalars found under {args.log_root}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(3, 2, figsize=(15, 14), constrained_layout=True)
    add_series(
        axes[0, 0], scalars,
        ["train/loss", "train/correspondence_loss"], args.smoothing,
    )
    axes[0, 0].set_title("Total and correspondence objectives")

    add_series(
        axes[0, 1], scalars,
        ["train/flow_loss", "train/detail_loss", "train/sigma_loss", "train/outside_velocity_loss"],
        args.smoothing,
    )
    axes[0, 1].set_title("Reconstruction/flow components")

    add_series(
        axes[1, 0], scalars,
        [
            "train/correspondence_nll",
            "train/correspondence_photometric",
            "train/correspondence_value",
            "train/correspondence_appearance_error",
        ],
        args.smoothing,
    )
    axes[1, 0].set_title("Correspondence loss components")

    add_series(
        axes[1, 1], scalars,
        [
            "train/correspondence_center_loss",
            "train/correspondence_entropy",
            "train/correspondence_target_mass",
            "train/correspondence_coverage",
        ],
        args.smoothing,
    )
    axes[1, 1].set_title("Routing quality (mass/coverage higher is better)")

    add_series(
        axes[2, 0], scalars,
        [
            "train/correspondence/coarse/target_mass",
            "train/correspondence/middle/target_mass",
            "train/correspondence/detail/target_mass",
        ],
        args.smoothing,
        labels={
            "train/correspondence/coarse/target_mass": "coarse",
            "train/correspondence/middle/target_mass": "middle",
            "train/correspondence/detail/target_mass": "detail",
        },
    )
    axes[2, 0].set_title("DINO-target attention mass by scale")

    gradients = aggregate_gradients(scalars)
    add_series(axes[2, 1], gradients, sorted(gradients), max(1, args.smoothing // 50))
    axes[2, 1].set_title("Mean garment-attention gradient norm by scale")

    figure.suptitle("VTON training diagnostics across resumed runs", fontsize=16)
    figure.savefig(output_dir / "loss_and_routing_curves.png", dpi=160)
    plt.close(figure)

    summary = {
        "event_files": event_files,
        "smoothing_points": args.smoothing,
        "recent_points": args.recent_points,
        "trends": trend_summary(scalars, args.recent_points),
    }
    with open(output_dir / "loss_trends.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoothing", type=int, default=200)
    parser.add_argument("--recent-points", type=int, default=500)
    main(parser.parse_args())

