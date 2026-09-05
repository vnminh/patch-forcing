#!/usr/bin/env python3
"""Visualize DINOv3 teacher correspondences and learned VTON garment attention."""

import argparse
import gc
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_DIR))

from jutils import instantiate_from_config

from patch_flow.correspondence import (
    DinoCorrespondenceTeacher,
    grid_coordinates,
    mask_to_token_valid,
    neighbourhood_mass,
)
from patch_flow.models.pf_transformer_vton import VTONPatchForcingDiT
from patch_flow.vae_features import encode_vae_pyramid
from patch_flow.vton_data import VTONHDDataset


QUERY_LOCATIONS = (
    ("left sleeve", 0.25, 0.34),
    ("left logo", 0.42, 0.36),
    ("right logo", 0.58, 0.36),
    ("right sleeve", 0.75, 0.34),
    ("left torso", 0.43, 0.52),
    ("right torso", 0.57, 0.52),
)


def image_array(tensor):
    return ((tensor.detach().float().cpu().permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)


def without_pretrained(config):
    config = OmegaConf.create(config)
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    config.params.pretrained_ckpt = None
    return config


def nearest_valid_queries(valid, locations):
    height, width = valid.shape
    candidates = torch.nonzero(valid, as_tuple=False)
    if candidates.numel() == 0:
        raise ValueError("The sample has no editable person tokens")
    selected = []
    for name, u, v in locations:
        desired = torch.tensor((v * height - 0.5, u * width - 0.5), device=candidates.device)
        index = ((candidates.float() - desired).square().sum(-1)).argmin()
        row, column = candidates[index].tolist()
        selected.append((name, int(row), int(column), int(row * width + column)))
    return selected


def teacher_pixels(teacher, images):
    size = teacher.resolve_input_size(images)
    pixels = F.interpolate(
        (images.float() + 1) / 2,
        size=size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return (pixels - teacher.image_mean) / teacher.image_std, size


@torch.no_grad()
def cls_attention(teacher, images):
    pixels, size = teacher_pixels(teacher, images)
    output = teacher.model(pixel_values=pixels, output_attentions=True)
    attentions = getattr(output, "attentions", None)
    if not attentions or attentions[-1] is None:
        return None
    height, width = size[0] // teacher.patch_size, size[1] // teacher.patch_size
    patch_count = height * width
    prefix = attentions[-1].shape[-1] - patch_count
    weights = attentions[-1][:, :, 0, prefix:].float().mean(1)
    return weights.reshape(weights.shape[0], height, width)


@torch.no_grad()
def teacher_diagnostics(teacher, person, garment, edit_mask, garment_mask, person_grid):
    person_features = teacher.features(person, person_grid)
    garment_features = teacher.features(garment, teacher.garment_grid)
    garment_grid = tuple(garment_features.shape[-2:])
    person_tokens = F.normalize(person_features.float().flatten(2).transpose(1, 2), dim=-1)
    garment_tokens = F.normalize(garment_features.float().flatten(2).transpose(1, 2), dim=-1)
    similarity_matrix = torch.einsum("bpc,bgc->bpg", person_tokens, garment_tokens)
    garment_valid = mask_to_token_valid(garment_mask, garment_grid)
    masked = similarity_matrix.masked_fill(~garment_valid[:, None, :], torch.finfo(torch.float32).min)
    best_similarity, best_index = masked.max(-1)
    coordinates = grid_coordinates(garment_grid, similarity_matrix.device, similarity_matrix.dtype)
    target = coordinates[best_index]
    person_valid = mask_to_token_valid(edit_mask, person_grid)
    weight = person_valid & (best_similarity >= 0.35)
    return {
        "person_features": person_features,
        "garment_features": garment_features,
        "garment_grid": garment_grid,
        "similarity_matrix": similarity_matrix,
        "masked_similarity": masked,
        "best_similarity": best_similarity,
        "best_index": best_index,
        "target": target,
        "weight": weight,
        "person_valid": person_valid,
        "garment_valid": garment_valid,
        "person_cls_attention": cls_attention(teacher, person),
        "garment_cls_attention": cls_attention(teacher, garment),
    }


def overlay(ax, image, heatmap, title, marker=None, cmap="magma"):
    ax.imshow(image)
    heatmap = np.asarray(heatmap, dtype=np.float32)
    finite = np.isfinite(heatmap)
    if finite.any():
        low, high = np.percentile(heatmap[finite], (5, 99))
        if high <= low:
            high = low + 1e-6
        masked = np.ma.masked_where(~finite, heatmap)
        ax.imshow(masked, cmap=cmap, alpha=0.62, interpolation="bilinear", vmin=low, vmax=high,
                  extent=(0, image.shape[1], image.shape[0], 0))
    if marker is not None:
        ax.scatter(marker[0], marker[1], marker="x", s=90, linewidth=2.5, c="cyan")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def save_teacher_figure(output, person, garment, diagnostics, queries):
    person_np, garment_np = image_array(person[0]), image_array(garment[0])
    person_grid = tuple(diagnostics["person_features"].shape[-2:])
    garment_grid = diagnostics["garment_grid"]
    rows = len(queries) + 1
    figure, axes = plt.subplots(rows, 3, figsize=(12, 3.3 * rows), constrained_layout=True)
    axes[0, 0].imshow(person_np)
    colors = plt.cm.tab10(np.linspace(0, 1, len(queries)))
    for color, (name, row, column, _) in zip(colors, queries):
        x = (column + 0.5) / person_grid[1] * person_np.shape[1]
        y = (row + 0.5) / person_grid[0] * person_np.shape[0]
        axes[0, 0].scatter(x, y, color=color, s=45)
        axes[0, 0].text(x + 4, y, name, color=color, fontsize=8, weight="bold")
    axes[0, 0].set_title("Person queries (editable region)")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(garment_np)
    for color, (_, _, _, query_index) in zip(colors, queries):
        u, v = diagnostics["target"][0, query_index].cpu().tolist()
        axes[0, 1].scatter(u * garment_np.shape[1], v * garment_np.shape[0], marker="x", color=color, s=55)
    axes[0, 1].set_title("DINOv3 pseudo-GT matches")
    axes[0, 1].axis("off")
    garment_attention = diagnostics["garment_cls_attention"]
    if garment_attention is not None:
        overlay(axes[0, 2], garment_np, garment_attention[0].cpu(), "DINOv3 garment CLS attention")
    else:
        axes[0, 2].text(0.5, 0.5, "CLS attention unavailable", ha="center")
        axes[0, 2].axis("off")

    for plot_row, (color, (name, row, column, query_index)) in enumerate(zip(colors, queries), start=1):
        query_image = person_np.copy()
        axes[plot_row, 0].imshow(query_image)
        qx = (column + 0.5) / person_grid[1] * person_np.shape[1]
        qy = (row + 0.5) / person_grid[0] * person_np.shape[0]
        axes[plot_row, 0].scatter(qx, qy, color=color, s=75)
        similarity = diagnostics["masked_similarity"][0, query_index].reshape(garment_grid).cpu().numpy()
        u, v = diagnostics["target"][0, query_index].cpu().tolist()
        marker = (u * garment_np.shape[1], v * garment_np.shape[0])
        confidence = diagnostics["best_similarity"][0, query_index].item()
        valid = bool(diagnostics["weight"][0, query_index])
        axes[plot_row, 0].set_title(f"{name}: person token ({row}, {column})")
        axes[plot_row, 0].axis("off")
        overlay(
            axes[plot_row, 1], garment_np, similarity,
            f"DINOv3 similarity; best={confidence:.3f}, supervised={valid}", marker,
        )
        flat = similarity.reshape(-1)
        finite = np.isfinite(flat)
        values = np.sort(flat[finite])[-10:][::-1]
        axes[plot_row, 2].bar(np.arange(len(values)), values, color=color)
        axes[plot_row, 2].axhline(0.35, color="red", linestyle="--", label="training threshold")
        axes[plot_row, 2].set_ylim(min(0, float(values.min()) - 0.05), 1)
        axes[plot_row, 2].set_title("Top-10 garment-token similarities")
        axes[plot_row, 2].legend(fontsize=7)
    figure.suptitle("DINOv3 teacher diagnostics for paired sample", fontsize=14)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def load_student(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    hyper_parameters = checkpoint.get("hyper_parameters")
    if hyper_parameters is None:
        raise ValueError("Checkpoint has no hyper_parameters")
    model = instantiate_from_config(without_pretrained(hyper_parameters["model"]))
    state = VTONPatchForcingDiT._select_checkpoint_state(checkpoint, use_ema=False)
    model.load_state_dict(state, strict=True)
    model.eval()
    first_stage = instantiate_from_config(hyper_parameters["first_stage"]).eval()
    flow = instantiate_from_config(hyper_parameters["flow"])
    del state, checkpoint
    gc.collect()
    return model, first_stage, flow


@torch.no_grad()
def student_attention(model, first_stage, flow, sample, timestep, seed):
    person = sample["person"]
    garment = sample["garment"]
    edit_mask = sample["agnostic_mask"]
    person_agnostic = sample["person_agnostic"]
    garment_mask = sample["garment_mask"]
    target, _, _ = encode_vae_pyramid(first_stage, person)
    agnostic = first_stage.encode(person_agnostic)
    garment_latent, garment_middle, garment_detail = encode_vae_pyramid(first_stage, garment)
    masks = flow.prepare_masks(edit_mask, target.shape[-2:], target.dtype)
    context = agnostic * (1 - masks.latent)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(target.shape, generator=generator, dtype=target.dtype)
    token_count = masks.token.shape[1]
    times = torch.full((1, token_count), float(timestep), dtype=target.dtype)
    xt, _, effective_times, masks = flow.get_interpolants(
        target, context, edit_mask.float(), x0=noise, t=times, masks=masks
    )
    label = torch.full((1,), model.y_embedder.num_classes, dtype=torch.long)
    _, _, maps = model(
        x=xt,
        t=effective_times,
        y=label,
        person_agnostic=context,
        person_mask=masks.condition,
        edit_mask=masks.condition,
        garment=garment_latent,
        garment_middle=garment_middle,
        garment_detail=garment_detail,
        garment_mask=garment_mask,
        return_uncertainty=True,
        return_garment_attention=True,
        garment_attention_scales=("coarse", "middle", "detail"),
    )
    return maps, masks.token


def aggregate_student_maps(attention_maps):
    by_scale = defaultdict(list)
    grids = {}
    for entry in attention_maps:
        weights = entry["weights"].float()
        if weights.ndim == 4:
            weights = weights.mean(dim=1)
        by_scale[entry["scale"]].append(weights)
        grids[entry["scale"]] = tuple(entry["grid"])
    return {scale: torch.stack(values).mean(0) for scale, values in by_scale.items()}, grids


def student_metrics(attention, grids, teacher, radii):
    weight = teacher["weight"].float()
    denominator = weight.sum().clamp_min(1)
    metrics = {}
    for scale, weights in attention.items():
        grid = grids[scale]
        coordinates = grid_coordinates(grid, weights.device, weights.dtype)
        center = weights @ coordinates
        center_error = (((center - teacher["target"]).square().sum(-1).sqrt()) * weight).sum() / denominator
        mass = neighbourhood_mass(weights, teacher["target"], grid, radii[scale])
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(-1) / np.log(weights.shape[-1])
        peak = coordinates[weights.argmax(-1)]
        peak_error = (((peak - teacher["target"]).square().sum(-1).sqrt()) * weight).sum() / denominator
        metrics[scale] = {
            "blocks": sum(1 for entry in teacher.get("attention_maps", []) if entry["scale"] == scale),
            "grid": list(grid),
            "target_mass": float((mass * weight).sum() / denominator),
            "center_error": float(center_error),
            "peak_error": float(peak_error),
            "normalized_entropy": float((entropy * weight).sum() / denominator),
        }
    return metrics


def save_student_figure(output, garment, teacher, queries, attention, grids):
    garment_np = image_array(garment[0])
    scales = [scale for scale in ("coarse", "middle", "detail") if scale in attention]
    figure, axes = plt.subplots(len(queries), len(scales) + 1, figsize=(4 * (len(scales) + 1), 3.5 * len(queries)),
                                constrained_layout=True)
    for row_index, (name, _, _, query_index) in enumerate(queries):
        teacher_grid = teacher["garment_grid"]
        teacher_heat = teacher["masked_similarity"][0, query_index].reshape(teacher_grid).cpu().numpy()
        target_u, target_v = teacher["target"][0, query_index].cpu().tolist()
        target_marker = (target_u * garment_np.shape[1], target_v * garment_np.shape[0])
        overlay(axes[row_index, 0], garment_np, teacher_heat, f"{name}: DINOv3 pseudo-GT", target_marker)
        for column_index, scale in enumerate(scales, start=1):
            heat = attention[scale][0, query_index].reshape(grids[scale]).cpu().numpy()
            peak = np.unravel_index(int(np.argmax(heat)), heat.shape)
            peak_xy = (
                (peak[1] + 0.5) / heat.shape[1] * garment_np.shape[1],
                (peak[0] + 0.5) / heat.shape[0] * garment_np.shape[0],
            )
            overlay(axes[row_index, column_index], garment_np, heat, f"learned {scale} attention", target_marker, "viridis")
            axes[row_index, column_index].scatter(*peak_xy, facecolors="none", edgecolors="lime", s=90, linewidth=2)
    figure.suptitle("DINOv3 target (cyan ×) versus learned attention peak (green ○)", fontsize=14)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main(args):
    torch.set_num_threads(args.cpu_threads)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = VTONHDDataset(
        args.dataset_root,
        split=args.split,
        image_size=(args.height, args.width),
        random_flip=False,
        paired=True,
        preview_sample_id=args.sample,
    )
    raw = dataset[0]
    sample = {key: value[None] if isinstance(value, torch.Tensor) else value for key, value in raw.items()}
    person, garment = sample["person"], sample["garment"]
    person_grid = (args.height // 16, args.width // 16)

    config = OmegaConf.load(args.config)
    trainer_params = config.trainer.params
    teacher = DinoCorrespondenceTeacher(
        model_name=trainer_params.correspondence_teacher_name,
        input_size=trainer_params.correspondence_teacher_input_size,
        garment_grid=trainer_params.correspondence_garment_grid,
    ).eval()
    if hasattr(teacher.model, "set_attn_implementation"):
        teacher.model.set_attn_implementation("eager")
    else:
        teacher.model.config._attn_implementation = "eager"
    diagnostics = teacher_diagnostics(
        teacher, person, garment, sample["agnostic_mask"], sample["garment_mask"], person_grid
    )
    queries = nearest_valid_queries(diagnostics["person_valid"][0].reshape(person_grid), QUERY_LOCATIONS)
    save_teacher_figure(output_dir / "dino_attention_and_correspondence.png", person, garment, diagnostics, queries)

    summary = {
        "sample": args.sample,
        "teacher": teacher.model_name,
        "person_grid": list(person_grid),
        "garment_grid": list(diagnostics["garment_grid"]),
        "teacher_coverage": float(diagnostics["weight"].float().sum() / diagnostics["person_valid"].float().sum()),
        "teacher_mean_similarity_editable": float(
            (diagnostics["best_similarity"] * diagnostics["person_valid"]).sum()
            / diagnostics["person_valid"].sum()
        ),
        "queries": [],
    }
    for name, row, column, index in queries:
        summary["queries"].append({
            "name": name,
            "person_token": [row, column],
            "dino_target_uv": diagnostics["target"][0, index].cpu().tolist(),
            "dino_similarity": float(diagnostics["best_similarity"][0, index]),
            "supervised": bool(diagnostics["weight"][0, index]),
        })

    if args.checkpoint:
        del teacher
        gc.collect()
        model, first_stage, flow = load_student(args.checkpoint)
        attention_maps, _ = student_attention(model, first_stage, flow, sample, args.timestep, args.seed)
        attention, grids = aggregate_student_maps(attention_maps)
        diagnostics["attention_maps"] = attention_maps
        radii = dict(trainer_params.correspondence_nll_radius)
        summary["checkpoint"] = args.checkpoint
        summary["timestep"] = args.timestep
        summary["learned_attention"] = student_metrics(attention, grids, diagnostics, radii)
        save_student_figure(
            output_dir / "learned_attention_vs_dino.png", garment, diagnostics, queries, attention, grids
        )

    with open(output_dir / "attention_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved diagnostics to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample", default="00055_00.jpg")
    parser.add_argument("--split", default="test")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--timestep", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--cpu-threads", type=int, default=12)
    main(parser.parse_args())
