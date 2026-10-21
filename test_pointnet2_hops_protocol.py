#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PointNet++ evaluation protocol for HOPS.

Designed for the analysis protocol:

                    HOPS
                     |
            +--------+---------+
            |                  |
      Labeled Val        Hidden-GT Test
        18 TLS          TLS/UAV/UGV/SFM
            |                  |
     quantitative            qualitative
       comparison          generalization

Validation outputs:
    - mIoU / mAcc / OA
    - per-class IoU / Precision / Recall / F1
    - Apple/Fruit IoU / Precision / Recall / F1
    - per-scene CSV + global JSON
    - semantic GT/prediction PLY
    - fruit error visualization PLY

Hidden-GT test outputs:
    - semantic prediction PLY
    - apple/fruit visualization PLY
    - input RGB PLY
    - prediction / confidence / vote_count NPY
    - NO quantitative metric is reported when GT is absent

Corrected HOPS train labels:
    0 ground
    1 trunk
    2 canopy
    3 apple
    4 pole

Raw semantic interpretation:
    raw 0 -> ignore
    raw 1 -> train 0 ground
    raw 2 -> train 1 trunk
    raw 3 -> train 2 canopy
    raw 4 -> train 3 apple
    raw 5 -> train 4 pole
    raw 6 -> ignore

Expected model-ready PointNet++ NPY:
    labeled   [N,7] = XYZ + RGB(0..255) + label(0..4)
    unlabeled [N,6] = XYZ + RGB(0..255)

Expected directory:
    pointnet_npy/
        val/
            TLS/
                *.npy
        test/
            TLS/
                *.npy
            UAV/
                *.npy
            UGV/
                *.npy
            SFM/
                *.npy

The script uses HOPSWholeSceneDataset, valid_mask and source indices to map
sliding-window predictions back to every original source point exactly.
Overlapping windows are fused by mean softmax probability.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import os
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from tqdm import tqdm

from data_utils.HOPSDataLoader_PointNet import HOPSWholeSceneDataset


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR
sys.path.append(str(ROOT_DIR / "models"))

NUM_CLASSES = 5
CLASS_NAMES = [
    "ground",
    "trunk",
    "canopy",
    "apple",
    "pole",
]
APPLE_CLASS = 3

SEMANTIC_COLORS = np.asarray(
    [
        [153, 102, 51],   # ground
        [112, 70, 35],    # trunk
        [46, 160, 67],    # canopy
        [220, 30, 30],    # apple
        [70, 110, 200],   # pole
    ],
    dtype=np.uint8,
)

# Fruit-centric visualization:
# other / TP / FP / FN
FRUIT_OTHER_COLOR = np.asarray([150, 150, 150], dtype=np.uint8)
FRUIT_TP_COLOR = np.asarray([40, 180, 70], dtype=np.uint8)
FRUIT_FP_COLOR = np.asarray([230, 40, 40], dtype=np.uint8)
FRUIT_FN_COLOR = np.asarray([40, 90, 230], dtype=np.uint8)
FRUIT_PRED_COLOR = np.asarray([220, 30, 30], dtype=np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(
        "PointNet++ HOPS validation + hidden-test evaluation"
    )

    parser.add_argument(
        "--protocol",
        choices=["val", "test", "all"],
        default="val",
        help=(
            "val: labeled TLS quantitative evaluation; "
            "test: hidden-GT TLS/UAV/UGV/SFM qualitative inference; "
            "all: run both."
        ),
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="D:/project_py/dataset/pointnet_npy",
        help="root containing val/ and test/ directories",
    )

    parser.add_argument(
        "--val_sensor",
        type=str,
        default="TLS",
    )

    parser.add_argument(
        "--test_sensors",
        nargs="+",
        default=["TLS", "UAV", "UGV", "SFM"],
        help="hidden-test sensor domains",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="experiment folder under ./log/sem_seg/",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="pointnet2_sem_seg_msg",
        help="PointNet++ model module",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help=(
            "optional checkpoint path; default "
            "./log/sem_seg/<log_dir>/checkpoints/best_model.pth"
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help=(
            "default: ./log/sem_seg/<log_dir>/hops_protocol_eval"
        ),
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="number of sliding-window blocks per inference batch",
    )

    parser.add_argument(
        "--num_point",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--block_size",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--stride",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--save_npz",
        action="store_true",
        help="also save a compressed all-in-one result.npz per scene",
    )

    parser.add_argument(
        "--no_ply",
        action="store_true",
        help="disable PLY exports",
    )

    return parser.parse_args()


def safe_div(numerator, denominator):
    return (
        float(numerator) / float(denominator)
        if denominator > 0
        else 0.0
    )


def confusion_matrix(gt, pred):
    gt = np.asarray(gt, dtype=np.int64).reshape(-1)
    pred = np.asarray(pred, dtype=np.int64).reshape(-1)

    if gt.shape != pred.shape:
        raise ValueError(
            f"GT/prediction shape mismatch: {gt.shape} vs {pred.shape}"
        )

    if not np.all(
        np.isin(np.unique(gt), np.arange(NUM_CLASSES))
    ):
        raise ValueError(
            f"GT contains invalid labels: {np.unique(gt).tolist()}"
        )

    if not np.all(
        np.isin(np.unique(pred), np.arange(NUM_CLASSES))
    ):
        raise ValueError(
            f"Prediction contains invalid labels: {np.unique(pred).tolist()}"
        )

    encoded = gt * NUM_CLASSES + pred

    return np.bincount(
        encoded,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(
        NUM_CLASSES,
        NUM_CLASSES,
    ).astype(np.int64)


def metrics_from_confusion(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)

    tp = np.diag(matrix).astype(np.float64)
    gt_count = matrix.sum(axis=1).astype(np.float64)
    pred_count = matrix.sum(axis=0).astype(np.float64)

    fp = pred_count - tp
    fn = gt_count - tp
    union = tp + fp + fn

    iou = np.divide(
        tp,
        union,
        out=np.zeros_like(tp),
        where=union > 0,
    )

    precision = np.divide(
        tp,
        pred_count,
        out=np.zeros_like(tp),
        where=pred_count > 0,
    )

    recall = np.divide(
        tp,
        gt_count,
        out=np.zeros_like(tp),
        where=gt_count > 0,
    )

    f1 = np.divide(
        2.0 * tp,
        2.0 * tp + fp + fn,
        out=np.zeros_like(tp),
        where=(2.0 * tp + fp + fn) > 0,
    )

    gt_present = gt_count > 0

    result = {
        "OA": safe_div(tp.sum(), matrix.sum()),
        "mAcc": (
            float(recall[gt_present].mean())
            if np.any(gt_present)
            else 0.0
        ),
        # Fixed five-class benchmark value.
        "mIoU": float(iou.mean()),
        "confusion_matrix": matrix.tolist(),
    }

    for class_id, class_name in enumerate(CLASS_NAMES):
        result[f"{class_name}_IoU"] = float(iou[class_id])
        result[f"{class_name}_Precision"] = float(precision[class_id])
        result[f"{class_name}_Recall"] = float(recall[class_id])
        result[f"{class_name}_F1"] = float(f1[class_id])
        result[f"{class_name}_GT_points"] = int(gt_count[class_id])
        result[f"{class_name}_Pred_points"] = int(pred_count[class_id])

    # Fruit == apple in the current HOPS five-class protocol.
    result["Fruit IoU"] = float(iou[APPLE_CLASS])
    result["Fruit Precision"] = float(precision[APPLE_CLASS])
    result["Fruit Recall"] = float(recall[APPLE_CLASS])
    result["Fruit F1"] = float(f1[APPLE_CLASS])

    return result


def normalize_model_output(model_output, expected_points):
    """
    Normalize to [B,N,C].
    """
    if isinstance(model_output, (tuple, list)):
        model_output = model_output[0]

    elif isinstance(model_output, dict):
        for key in ("logits", "pred", "seg_pred", "output"):
            if key in model_output:
                model_output = model_output[key]
                break
        else:
            raise KeyError(
                "No segmentation tensor in model output dictionary"
            )

    if (
        not torch.is_tensor(model_output)
        or model_output.ndim != 3
    ):
        raise ValueError(
            f"Expected model output [B,N,C] or [B,C,N], "
            f"got {type(model_output)} / "
            f"{getattr(model_output, 'shape', None)}"
        )

    if (
        model_output.shape[1] == expected_points
        and model_output.shape[2] == NUM_CLASSES
    ):
        return model_output

    if (
        model_output.shape[1] == NUM_CLASSES
        and model_output.shape[2] == expected_points
    ):
        return model_output.transpose(
            1,
            2,
        ).contiguous()

    raise ValueError(
        f"Unexpected model output shape {tuple(model_output.shape)}; "
        f"expected N={expected_points}, C={NUM_CLASSES}"
    )


def load_classifier(
    model_name,
    checkpoint_path,
    device,
):
    model_module = importlib.import_module(
        model_name
    )

    classifier = model_module.get_model(
        NUM_CLASSES
    ).to(
        device
    )

    try:
        checkpoint = torch.load(
            str(checkpoint_path),
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            str(checkpoint_path),
            map_location=device,
        )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Unsupported checkpoint type: {type(checkpoint)}"
        )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif (
        "model" in checkpoint
        and isinstance(checkpoint["model"], dict)
    ):
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    state_dict = {
        str(key).removeprefix("module."): value
        for key, value in state_dict.items()
    }

    classifier.load_state_dict(
        state_dict,
        strict=True,
    )

    return classifier.eval()


def resolve_sensor_directory(
    data_root,
    split,
    sensor,
):
    split_dir = Path(data_root) / split

    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Missing split directory: {split_dir}"
        )

    direct = split_dir / sensor

    if direct.is_dir():
        return direct

    for child in split_dir.iterdir():
        if (
            child.is_dir()
            and child.name.lower() == sensor.lower()
        ):
            return child

    raise FileNotFoundError(
        f"Sensor '{sensor}' not found under {split_dir}. "
        f"Available: {[p.name for p in split_dir.iterdir() if p.is_dir()]}"
    )


def discover_scene_files(
    data_root,
    split,
    sensor,
    require_labels,
):
    sensor_dir = resolve_sensor_directory(
        data_root,
        split,
        sensor,
    )

    files = sorted(
        path.resolve()
        for path in sensor_dir.rglob("*.npy")
        if path.is_file()
    )

    if not files:
        raise FileNotFoundError(
            f"No NPY files under {sensor_dir}"
        )

    valid_files = []

    for path in files:
        array = np.load(
            path,
            mmap_mode="r",
            allow_pickle=False,
        )

        if (
            array.ndim != 2
            or array.shape[1] not in (6, 7)
        ):
            print(
                f"[SKIP] {path}: expected [N,6] or [N,7], "
                f"got {array.shape}"
            )
            continue

        if require_labels and array.shape[1] != 7:
            print(
                f"[SKIP] {path}: labeled validation requires [N,7]"
            )
            continue

        if array.shape[1] == 7:
            labels = np.asarray(
                array[:, 6],
                dtype=np.int64,
            )

            if not np.all(
                np.isin(
                    np.unique(labels),
                    np.arange(NUM_CLASSES),
                )
            ):
                print(
                    f"[SKIP] {path}: invalid labels "
                    f"{np.unique(labels).tolist()}"
                )
                continue

        valid_files.append(
            path
        )

    if not valid_files:
        raise RuntimeError(
            f"No valid scenes remain under {sensor_dir}"
        )

    return valid_files


def write_runtime_split(
    files,
    output_dir,
    split,
    sensor,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        output_dir
        / f"runtime_{split}_{sensor.lower()}_pointnet.txt"
    )

    path.write_text(
        "\n".join(
            str(file_path)
            for file_path in files
        )
        + "\n",
        encoding="utf-8",
    )

    return path


def normalize_rgb(rgb):
    rgb = np.asarray(
        rgb,
        dtype=np.float32,
    )

    if rgb.size == 0:
        return np.zeros(
            (0, 3),
            dtype=np.uint8,
        )

    if float(
        np.nanmax(
            rgb
        )
    ) <= 1.5:
        rgb = rgb * 255.0

    return np.clip(
        rgb,
        0,
        255,
    ).astype(
        np.uint8
    )


def export_rgb_ply(
    path,
    xyz,
    rgb,
):
    xyz = np.asarray(
        xyz,
        dtype=np.float32,
    )

    rgb = np.asarray(
        rgb,
        dtype=np.uint8,
    )

    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or rgb.shape != xyz.shape
    ):
        raise ValueError(
            f"PLY shape mismatch: xyz={xyz.shape}, rgb={rgb.shape}"
        )

    vertex = np.empty(
        xyz.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )

    vertex["x"] = xyz[:, 0]
    vertex["y"] = xyz[:, 1]
    vertex["z"] = xyz[:, 2]

    vertex["red"] = rgb[:, 0]
    vertex["green"] = rgb[:, 1]
    vertex["blue"] = rgb[:, 2]

    PlyData(
        [
            PlyElement.describe(
                vertex,
                "vertex",
            )
        ],
        text=False,
    ).write(
        str(path)
    )


def semantic_colors(labels):
    labels = np.asarray(
        labels,
        dtype=np.int64,
    ).reshape(-1)

    return SEMANTIC_COLORS[
        labels
    ]


def fruit_prediction_colors(pred):
    pred = np.asarray(
        pred,
        dtype=np.int64,
    ).reshape(-1)

    colors = np.tile(
        FRUIT_OTHER_COLOR[
            None,
            :
        ],
        (
            pred.shape[0],
            1,
        ),
    )

    colors[
        pred == APPLE_CLASS
    ] = FRUIT_PRED_COLOR

    return colors.astype(
        np.uint8
    )


def fruit_error_colors(
    gt,
    pred,
):
    gt = np.asarray(
        gt,
        dtype=np.int64,
    ).reshape(-1)

    pred = np.asarray(
        pred,
        dtype=np.int64,
    ).reshape(-1)

    gt_apple = (
        gt == APPLE_CLASS
    )

    pred_apple = (
        pred == APPLE_CLASS
    )

    colors = np.tile(
        FRUIT_OTHER_COLOR[
            None,
            :
        ],
        (
            gt.shape[0],
            1,
        ),
    )

    colors[
        gt_apple
        & pred_apple
    ] = FRUIT_TP_COLOR

    colors[
        ~gt_apple
        & pred_apple
    ] = FRUIT_FP_COLOR

    colors[
        gt_apple
        & ~pred_apple
    ] = FRUIT_FN_COLOR

    return colors.astype(
        np.uint8
    )


def save_scene_artifacts(
    scene_dir,
    points,
    pred,
    confidence,
    vote_count,
    gt,
    save_ply,
    save_npz,
):
    scene_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    points = np.asarray(
        points,
        dtype=np.float32,
    )

    xyz = points[:, :3]
    rgb = normalize_rgb(
        points[:, 3:6]
    )

    np.save(
        scene_dir / "prediction.npy",
        pred,
        allow_pickle=False,
    )

    np.save(
        scene_dir / "confidence.npy",
        confidence,
        allow_pickle=False,
    )

    np.save(
        scene_dir / "vote_count.npy",
        vote_count,
        allow_pickle=False,
    )

    if gt is not None:
        np.save(
            scene_dir / "ground_truth.npy",
            gt,
            allow_pickle=False,
        )

    if save_npz:
        payload = {
            "xyz": xyz.astype(np.float32),
            "rgb": rgb,
            "prediction": pred.astype(np.int64),
            "confidence": confidence.astype(np.float32),
            "vote_count": vote_count.astype(np.int64),
            "source_index": np.arange(
                xyz.shape[0],
                dtype=np.int64,
            ),
        }

        if gt is not None:
            payload[
                "ground_truth"
            ] = gt.astype(
                np.int64
            )

        np.savez_compressed(
            scene_dir / "result.npz",
            **payload,
        )

    if not save_ply:
        return

    export_rgb_ply(
        scene_dir / "input_rgb.ply",
        xyz,
        rgb,
    )

    export_rgb_ply(
        scene_dir / "semantic_prediction.ply",
        xyz,
        semantic_colors(
            pred
        ),
    )

    export_rgb_ply(
        scene_dir / "fruit_prediction.ply",
        xyz,
        fruit_prediction_colors(
            pred
        ),
    )

    apple_mask = (
        pred == APPLE_CLASS
    )

    if np.any(
        apple_mask
    ):
        export_rgb_ply(
            scene_dir / "predicted_fruit_points.ply",
            xyz[
                apple_mask
            ],
            np.tile(
                FRUIT_PRED_COLOR[
                    None,
                    :
                ],
                (
                    int(
                        apple_mask.sum()
                    ),
                    1,
                ),
            ),
        )

    if gt is not None:
        export_rgb_ply(
            scene_dir / "semantic_ground_truth.ply",
            xyz,
            semantic_colors(
                gt
            ),
        )

        export_rgb_ply(
            scene_dir / "fruit_error_map.ply",
            xyz,
            fruit_error_colors(
                gt,
                pred,
            ),
        )


def run_whole_scene_inference(
    classifier,
    dataset,
    scene_index,
    batch_size,
    num_point,
    device,
):
    (
        scene_data,
        _scene_label_blocks,
        _scene_smpw,
        scene_point_index,
        valid_mask,
    ) = dataset[
        scene_index
    ]

    source_points = np.asarray(
        dataset.scene_points_list[
            scene_index
        ],
        dtype=np.float32,
    )

    source_gt = (
        dataset.semantic_labels_list[
            scene_index
        ]
    )

    source_count = int(
        source_points.shape[0]
    )

    probability_sum = np.zeros(
        (
            source_count,
            NUM_CLASSES,
        ),
        dtype=np.float64,
    )

    vote_count = np.zeros(
        source_count,
        dtype=np.int64,
    )

    num_blocks = int(
        scene_data.shape[0]
    )

    if scene_data.shape[1:] != (
        num_point,
        9,
    ):
        raise RuntimeError(
            f"Unexpected whole-scene block shape {scene_data.shape}; "
            f"expected [K,{num_point},9]"
        )

    for start in tqdm(
        range(
            0,
            num_blocks,
            batch_size,
        ),
        desc="blocks",
        leave=False,
    ):
        end = min(
            start + batch_size,
            num_blocks,
        )

        batch_data = np.asarray(
            scene_data[
                start:end
            ],
            dtype=np.float32,
        )

        batch_index = np.asarray(
            scene_point_index[
                start:end
            ],
            dtype=np.int64,
        )

        batch_valid = np.asarray(
            valid_mask[
                start:end
            ],
            dtype=bool,
        )

        torch_data = torch.from_numpy(
            batch_data
        ).float().to(
            device,
            non_blocking=True,
        ).transpose(
            2,
            1,
        ).contiguous()

        model_output = classifier(
            torch_data
        )

        logits = normalize_model_output(
            model_output,
            expected_points=num_point,
        )

        probability = torch.softmax(
            logits,
            dim=2,
        ).detach().cpu().numpy()

        for local_index in range(
            end - start
        ):
            real_mask = batch_valid[
                local_index
            ]

            source_indices = batch_index[
                local_index
            ][
                real_mask
            ]

            source_probability = probability[
                local_index
            ][
                real_mask
            ]

            np.add.at(
                probability_sum,
                source_indices,
                source_probability,
            )

            np.add.at(
                vote_count,
                source_indices,
                1,
            )

    missing = np.flatnonzero(
        vote_count == 0
    )

    if missing.size:
        raise RuntimeError(
            f"{missing.size}/{source_count} source points received "
            "no prediction. Check block_size/stride."
        )

    mean_probability = (
        probability_sum
        / vote_count[
            :,
            None
        ]
    )

    pred = np.argmax(
        mean_probability,
        axis=1,
    ).astype(
        np.int64
    )

    confidence = np.max(
        mean_probability,
        axis=1,
    ).astype(
        np.float32
    )

    if source_gt is not None:
        source_gt = np.asarray(
            source_gt,
            dtype=np.int64,
        ).reshape(-1)

    return (
        source_points,
        source_gt,
        pred,
        confidence,
        vote_count,
        mean_probability,
    )


def log_metrics(
    log_string,
    prefix,
    metrics,
):
    log_string(
        (
            f"{prefix}: "
            f"mIoU={metrics['mIoU']:.4f}, "
            f"mAcc={metrics['mAcc']:.4f}, "
            f"OA={metrics['OA']:.4f}, "
            f"Fruit IoU={metrics['Fruit IoU']:.4f}, "
            f"Fruit P={metrics['Fruit Precision']:.4f}, "
            f"Fruit R={metrics['Fruit Recall']:.4f}, "
            f"Fruit F1={metrics['Fruit F1']:.4f}"
        )
    )

    for class_name in CLASS_NAMES:
        log_string(
            (
                f"  {class_name:<7s} "
                f"IoU={metrics[f'{class_name}_IoU']:.4f}, "
                f"P={metrics[f'{class_name}_Precision']:.4f}, "
                f"R={metrics[f'{class_name}_Recall']:.4f}, "
                f"F1={metrics[f'{class_name}_F1']:.4f}, "
                f"GT={metrics[f'{class_name}_GT_points']}, "
                f"Pred={metrics[f'{class_name}_Pred_points']}"
            )
        )


def write_csv(
    path,
    rows,
):
    if not rows:
        return

    fieldnames = []

    for row in rows:
        for key in row:
            if (
                key != "confusion_matrix"
                and key not in fieldnames
            ):
                fieldnames.append(
                    key
                )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: value
                    for key, value in row.items()
                    if key in fieldnames
                }
            )


def evaluate_domain(
    classifier,
    args,
    split,
    sensor,
    require_labels,
    compute_metrics,
    output_root,
    device,
):
    domain_dir = (
        output_root
        / split
        / sensor
    )

    domain_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = discover_scene_files(
        data_root=args.data_root,
        split=split,
        sensor=sensor,
        require_labels=require_labels,
    )

    runtime_split = write_runtime_split(
        files=files,
        output_dir=domain_dir,
        split=split,
        sensor=sensor,
    )

    dataset = HOPSWholeSceneDataset(
        split_file=runtime_split,
        block_points=args.num_point,
        stride=args.stride,
        block_size=args.block_size,
    )

    print(
        f"\n========== {split.upper()} / {sensor} =========="
    )
    print(
        f"Scenes: {len(dataset)}"
    )

    global_confusion = np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES,
        ),
        dtype=np.int64,
    )

    per_scene_rows = []

    for scene_index, scene_path in enumerate(
        dataset.paths
    ):
        scene_name = Path(
            scene_path
        ).stem

        print(
            f"[{scene_index + 1}/{len(dataset)}] {scene_name}"
        )

        (
            source_points,
            source_gt,
            pred,
            confidence,
            vote_count,
            mean_probability,
        ) = run_whole_scene_inference(
            classifier=classifier,
            dataset=dataset,
            scene_index=scene_index,
            batch_size=args.batch_size,
            num_point=args.num_point,
            device=device,
        )

        scene_dir = (
            domain_dir
            / "scenes"
            / scene_name
        )

        save_scene_artifacts(
            scene_dir=scene_dir,
            points=source_points,
            pred=pred,
            confidence=confidence,
            vote_count=vote_count,
            gt=(
                source_gt
                if compute_metrics
                else None
            ),
            save_ply=(
                not args.no_ply
            ),
            save_npz=args.save_npz,
        )
        np.save(
            scene_dir / "class_probability.npy",
            mean_probability.astype(np.float32),
            allow_pickle=False,
        )

        if compute_metrics:
            if source_gt is None:
                raise RuntimeError(
                    f"{scene_name}: validation scene unexpectedly has no GT"
                )

            scene_confusion = confusion_matrix(
                source_gt,
                pred,
            )

            global_confusion += scene_confusion

            scene_metrics = metrics_from_confusion(
                scene_confusion
            )

            row = {
                "scene": scene_name,
                "split": split,
                "sensor": sensor,
                "points": int(
                    source_points.shape[0]
                ),
                **scene_metrics,
            }

            per_scene_rows.append(
                row
            )

            (
                scene_dir
                / "metrics.json"
            ).write_text(
                json.dumps(
                    scene_metrics,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            log_metrics(
                print,
                scene_name,
                scene_metrics,
            )

        else:
            # Hidden-GT test: prediction/visualization only.
            print(
                f"[QUAL] {scene_name}: semantic/fruit predictions saved; "
                "no hidden-test metric computed."
            )

    summary = {
        "split": split,
        "sensor": sensor,
        "scene_count": len(
            dataset
        ),
        "class_names": CLASS_NAMES,
        "fruit_class": "apple",
        "fruit_train_id": APPLE_CLASS,
        "block_size": float(
            args.block_size
        ),
        "stride": float(
            args.stride
        ),
        "num_point": int(
            args.num_point
        ),
        "evaluation_type": (
            "quantitative_labeled"
            if compute_metrics
            else "qualitative_hidden_gt"
        ),
    }

    if compute_metrics:
        global_metrics = metrics_from_confusion(
            global_confusion
        )

        summary.update(
            global_metrics
        )

        print(
            f"\n========== GLOBAL {split.upper()} / {sensor} =========="
        )

        log_metrics(
            print,
            "GLOBAL",
            global_metrics,
        )

        write_csv(
            domain_dir
            / "per_scene_metrics.csv",
            per_scene_rows,
        )

    (
        domain_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return summary


def main():
    args = parse_args()

    os.environ[
        "CUDA_VISIBLE_DEVICES"
    ] = args.gpu

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available"
        )

    device = torch.device(
        "cuda"
    )

    experiment_dir = (
        ROOT_DIR
        / "log"
        / "sem_seg"
        / args.log_dir
    )

    checkpoint_path = (
        Path(
            args.checkpoint
        ).expanduser().resolve()
        if args.checkpoint
        else (
            experiment_dir
            / "checkpoints"
            / "best_model.pth"
        )
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    output_root = (
        Path(
            args.output_dir
        ).expanduser().resolve()
        if args.output_dir
        else (
            experiment_dir
            / "hops_protocol_eval"
        ).resolve()
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    classifier = load_classifier(
        model_name=args.model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    print(
        "HOPS class mapping:"
    )

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):
        print(
            f"  {class_id}: {class_name}"
        )

    print(
        f"Fruit/Apple class ID: {APPLE_CLASS}"
    )

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    protocol_summary = {
        "checkpoint": str(
            checkpoint_path
        ),
        "model": args.model,
        "class_names": CLASS_NAMES,
        "fruit_train_id": APPLE_CLASS,
        "protocol": args.protocol,
        "domains": {},
    }

    if args.protocol in (
        "val",
        "all",
    ):
        val_summary = evaluate_domain(
            classifier=classifier,
            args=args,
            split="val",
            sensor=args.val_sensor,
            require_labels=True,
            compute_metrics=True,
            output_root=output_root,
            device=device,
        )

        protocol_summary[
            "domains"
        ][
            f"val/{args.val_sensor}"
        ] = val_summary

    if args.protocol in (
        "test",
        "all",
    ):
        for sensor in args.test_sensors:
            test_summary = evaluate_domain(
                classifier=classifier,
                args=args,
                split="test",
                sensor=sensor,
                require_labels=False,
                # Explicitly suppress hidden-GT metrics even if a test file
                # unexpectedly contains a label column.
                compute_metrics=False,
                output_root=output_root,
                device=device,
            )

            protocol_summary[
                "domains"
            ][
                f"test/{sensor}"
            ] = test_summary

    (
        output_root
        / "protocol_summary.json"
    ).write_text(
        json.dumps(
            protocol_summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\n[DONE] HOPS protocol results: {output_root}"
    )


if __name__ == "__main__":
    main()
