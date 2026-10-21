"""
HOPS semantic segmentation training for PointNet++.

Based on the user's current train_semseg.py, with the following fixes:
1. Corrected HOPS class order is:
       0 ground, 1 trunk, 2 canopy, 3 apple, 4 pole
2. Removes the second parse_args() call inside main().
3. Fixes checkpoint path construction and PyTorch >= 2.6 loading.
4. Fixes the validation class-frequency logging bug:
       labelweights[l - 1] -> class_distribution[l]
5. Prints and validates the REAL train/val label distribution before training.
6. Uses drop_last=False for validation.
7. Reports both:
       mIoU_5class   : fixed average over all five HOPS classes
       mIoU_present  : average only over GT-present classes
8. Supports either split txt files OR direct train/val directories.
9. Keeps the original PointNet++ training/evaluation logic as much as possible.

Expected HOPS v4 NPY:
    [N,7] = XYZ + RGB(0..255) + train_label(0..4)

Corrected HOPS mapping:
    raw 0 -> ignore
    raw 1 -> train 0 ground
    raw 2 -> train 1 trunk
    raw 3 -> train 2 canopy
    raw 4 -> train 3 apple
    raw 5 -> train 4 pole
    raw 6 -> ignore
"""

import argparse
import datetime
import importlib
import logging
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

import provider
from data_utils.HOPSDataLoader_PointNet import HOPSPointNetDataset


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "models"))

NUM_CLASSES = 5
CLASSES = [
    "ground",
    "trunk",
    "canopy",
    "apple",
    "pole",
]

CLASS2LABEL = {
    cls: index
    for index, cls in enumerate(CLASSES)
}

SEG_LABEL_TO_CAT = {
    index: cls
    for index, cls in enumerate(CLASSES)
}


def inplace_relu(module):
    classname = module.__class__.__name__
    if "ReLU" in classname:
        module.inplace = True


def compute_kan_regularization(model, lambda_reg=1e-4):
    reg_loss = 0
    for name, param in model.named_parameters():
        if "coeffs" in name:
            reg_loss += torch.norm(param, p=1)
    return lambda_reg * reg_loss


def parse_args():
    parser = argparse.ArgumentParser(
        "PointNet++ HOPS semantic segmentation training"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="pointnet2_sem_seg_KAN",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        choices=["Adam", "SGD"],
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--decay_rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--npoint",
        type=int,
        default=4096,
        help="points sampled per training block [default: 4096]",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--lr_decay",
        type=float,
        default=0.7,
    )

    # Prefer split files when they exist.
    parser.add_argument(
        "--train_split",
        type=str,
        default="./splits/train_pointnet.txt",
    )
    parser.add_argument(
        "--val_split",
        type=str,
        default="./splits/val_pointnet.txt",
    )

    # If split files do not exist, build temporary split files from these roots.
    parser.add_argument(
        "--train_root",
        type=str,
        default="D:/project_py/dataset/pointnet_npy/train",
        help="used when --train_split does not exist",
    )
    parser.add_argument(
        "--val_root",
        type=str,
        default="D:/project_py/dataset/pointnet_npy/val",
        help="used when --val_split does not exist",
    )

    parser.add_argument(
        "--block_size",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    return parser.parse_args()


def discover_scene_files(root):
    root = Path(root).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(
            f"HOPS data root does not exist: {root}"
        )

    if root.is_file():
        candidates = [root]
    else:
        candidates = sorted(
            root.rglob("*.npy")
        )

    valid = []

    for path in candidates:
        try:
            array = np.load(
                path,
                mmap_mode="r",
                allow_pickle=False,
            )
        except Exception:
            continue

        if (
            array.ndim == 2
            and array.shape[1] == 7
        ):
            labels = np.asarray(
                array[:, 6],
                dtype=np.int64,
            )

            unique = np.unique(labels)

            if np.all(
                np.isin(
                    unique,
                    np.arange(NUM_CLASSES),
                )
            ):
                valid.append(
                    path.resolve()
                )

    if not valid:
        raise RuntimeError(
            f"No labeled HOPS PointNet++ [N,7] scene files found under {root}"
        )

    return valid


def resolve_split_file(
    split_file,
    data_root,
    generated_path,
    split_name,
):
    """
    Use an existing split txt if present; otherwise auto-generate one from
    a directory of model-ready HOPS PointNet++ NPY files.
    """
    if split_file:
        split_path = Path(
            split_file
        ).expanduser()

        if split_path.is_file():
            return split_path.resolve()

    scene_files = discover_scene_files(
        data_root
    )

    generated_path = Path(
        generated_path
    )

    generated_path.write_text(
        "\n".join(
            str(path)
            for path in scene_files
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"[AUTO SPLIT] {split_name}: {len(scene_files)} scene(s)"
    )
    print(
        f"[AUTO SPLIT] generated {generated_path}"
    )

    return generated_path.resolve()


def dataset_label_statistics(dataset):
    """
    Count labels directly from the loaded scene labels, before block sampling.
    """
    counts = np.zeros(
        NUM_CLASSES,
        dtype=np.int64,
    )

    scene_records = []

    for path, labels in zip(
        dataset.paths,
        dataset.scene_labels,
    ):
        labels = np.asarray(
            labels,
            dtype=np.int64,
        ).reshape(-1)

        unique, class_counts = np.unique(
            labels,
            return_counts=True,
        )

        invalid = unique[
            ~np.isin(
                unique,
                np.arange(NUM_CLASSES),
            )
        ]

        if invalid.size:
            raise ValueError(
                f"{path}: invalid train labels {invalid.tolist()}"
            )

        scene_count = np.bincount(
            labels,
            minlength=NUM_CLASSES,
        ).astype(
            np.int64
        )

        counts += scene_count

        scene_records.append(
            (
                str(path),
                scene_count,
            )
        )

    total = int(
        counts.sum()
    )

    frequency = (
        counts.astype(
            np.float64
        )
        / max(
            total,
            1,
        )
    )

    return counts, frequency, scene_records


def log_dataset_distribution(
    log_string,
    name,
    dataset,
):
    counts, frequency, scene_records = (
        dataset_label_statistics(
            dataset
        )
    )

    log_string(
        f"========== {name} LABEL DISTRIBUTION =========="
    )

    log_string(
        f"{name} scene files: {len(dataset.paths)}"
    )

    log_string(
        f"{name} sampled dataset length: {len(dataset)}"
    )

    log_string(
        f"{name} source points: {int(counts.sum())}"
    )

    for class_id, class_name in enumerate(
        CLASSES
    ):
        log_string(
            (
                f"  {class_id} {class_name:<7s}: "
                f"{int(counts[class_id])} "
                f"({frequency[class_id] * 100.0:.4f}%)"
            )
        )

    # Important diagnostics.
    absent = [
        CLASSES[index]
        for index in range(NUM_CLASSES)
        if counts[index] == 0
    ]

    if absent:
        log_string(
            f"[WARN] {name} has ZERO points for classes: {absent}"
        )

    dominant_index = int(
        np.argmax(
            counts
        )
    )

    dominant_ratio = float(
        frequency[
            dominant_index
        ]
    )

    if dominant_ratio > 0.95:
        log_string(
            (
                f"[WARN] {name} is extremely imbalanced: "
                f"{CLASSES[dominant_index]} = "
                f"{dominant_ratio * 100.0:.3f}%"
            )
        )

    log_string(
        f"{name} loader loss weights: "
        + ", ".join(
            (
                f"{CLASSES[index]}="
                f"{float(dataset.labelweights[index]):.4f}"
            )
            for index in range(NUM_CLASSES)
        )
    )

    return counts, frequency


def compute_validation_metrics(
    total_correct_class,
    total_seen_class,
    total_iou_deno_class,
):
    correct = np.asarray(
        total_correct_class,
        dtype=np.float64,
    )

    seen = np.asarray(
        total_seen_class,
        dtype=np.float64,
    )

    union = np.asarray(
        total_iou_deno_class,
        dtype=np.float64,
    )

    iou = np.divide(
        correct,
        union,
        out=np.zeros_like(correct),
        where=union > 0,
    )

    acc = np.divide(
        correct,
        seen,
        out=np.zeros_like(correct),
        where=seen > 0,
    )

    # Fixed benchmark-style 5-class average.
    miou_5class = float(
        iou.mean()
    )

    # Diagnostic mean over classes actually present in GT.
    present = seen > 0

    miou_present = (
        float(
            iou[
                present
            ].mean()
        )
        if np.any(
            present
        )
        else 0.0
    )

    macc_present = (
        float(
            acc[
                present
            ].mean()
        )
        if np.any(
            present
        )
        else 0.0
    )

    return (
        iou,
        acc,
        miou_5class,
        miou_present,
        macc_present,
    )


def load_checkpoint_if_available(
    classifier,
    optimizer,
    checkpoint_path,
    log_string,
):
    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.is_file():
        log_string(
            "No existing model, starting training from scratch..."
        )
        return 0, 0.0, False

    try:
        checkpoint = torch.load(
            str(
                checkpoint_path
            ),
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            str(
                checkpoint_path
            ),
            map_location="cpu",
        )

    classifier.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    if (
        optimizer is not None
        and "optimizer_state_dict"
        in checkpoint
    ):
        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

    start_epoch = int(
        checkpoint.get(
            "epoch",
            -1,
        )
    ) + 1

    best_iou = float(
        checkpoint.get(
            "class_avg_iou",
            0.0,
        )
    )

    log_string(
        f"Use pretrained model: {checkpoint_path}"
    )
    log_string(
        f"Resume epoch: {start_epoch}"
    )
    log_string(
        f"Restored best mIoU: {best_iou:.6f}"
    )

    return (
        start_epoch,
        best_iou,
        True,
    )


def main(args):
    os.environ[
        "CUDA_VISIBLE_DEVICES"
    ] = args.gpu

    timestr = datetime.datetime.now().strftime(
        "%Y-%m-%d_%H-%M"
    )

    experiment_dir = Path(
        "./log/sem_seg"
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.log_dir is None:
        experiment_dir = (
            experiment_dir
            / timestr
        )
    else:
        experiment_dir = (
            experiment_dir
            / args.log_dir
        )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoints_dir = (
        experiment_dir
        / "checkpoints"
    )

    checkpoints_dir.mkdir(
        exist_ok=True
    )

    log_dir = (
        experiment_dir
        / "logs"
    )

    log_dir.mkdir(
        exist_ok=True
    )

    logger = logging.getLogger(
        "Model"
    )

    logger.setLevel(
        logging.INFO
    )

    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(
        log_dir
        / f"{args.model}.txt",
        encoding="utf-8",
    )

    file_handler.setLevel(
        logging.INFO
    )

    file_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        file_handler
    )

    def log_string(message):
        logger.info(
            str(
                message
            )
        )
        print(
            message
        )

    log_string(
        "PARAMETER ..."
    )

    log_string(
        args
    )

    log_string(
        "Corrected HOPS class mapping:"
    )

    for class_id, class_name in enumerate(
        CLASSES
    ):
        log_string(
            f"  train label {class_id}: {class_name}"
        )

    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    # ------------------------------------------------------------------
    # Resolve train / validation scene lists.
    # ------------------------------------------------------------------
    train_split_file = resolve_split_file(
        split_file=args.train_split,
        data_root=args.train_root,
        generated_path=(
            experiment_dir
            / "runtime_train_pointnet.txt"
        ),
        split_name="train",
    )

    val_split_file = resolve_split_file(
        split_file=args.val_split,
        data_root=args.val_root,
        generated_path=(
            experiment_dir
            / "runtime_val_pointnet.txt"
        ),
        split_name="val",
    )

    log_string(
        f"Train scene list: {train_split_file}"
    )

    log_string(
        f"Val scene list: {val_split_file}"
    )

    train_dataset = HOPSPointNetDataset(
        split_file=train_split_file,
        num_point=NUM_POINT,
        block_size=args.block_size,
        sample_rate=args.sample_rate,
    )

    val_dataset = HOPSPointNetDataset(
        split_file=val_split_file,
        num_point=NUM_POINT,
        block_size=args.block_size,
        sample_rate=args.sample_rate,
    )

    # Print raw source-scene distribution BEFORE stochastic block sampling.
    train_counts, train_frequency = log_dataset_distribution(
        log_string,
        "TRAIN",
        train_dataset,
    )

    val_counts, val_frequency = log_dataset_distribution(
        log_string,
        "VAL",
        val_dataset,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=lambda worker_id: np.random.seed(
            worker_id
            + int(
                time.time()
            )
        ),
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    weights = torch.tensor(
        train_dataset.labelweights,
        dtype=torch.float32,
        device="cuda",
    )

    log_string(
        "The number of training samples is: %d"
        % len(
            train_dataset
        )
    )

    log_string(
        "The number of validation samples is: %d"
        % len(
            val_dataset
        )
    )

    # ------------------------------------------------------------------
    # Model.
    # ------------------------------------------------------------------
    MODEL = importlib.import_module(
        args.model
    )

    shutil.copy(
        f"models/{args.model}.py",
        str(
            experiment_dir
        ),
    )

    shutil.copy(
        "models/pointnet2_utils.py",
        str(
            experiment_dir
        ),
    )

    classifier = MODEL.get_model(
        NUM_CLASSES
    ).cuda()

    criterion = MODEL.get_loss().cuda()

    classifier.apply(
        inplace_relu
    )

    def weights_init(module):
        classname = (
            module.__class__.__name__
        )

        if "Conv2d" in classname:
            torch.nn.init.xavier_normal_(
                module.weight.data
            )

            if module.bias is not None:
                torch.nn.init.constant_(
                    module.bias.data,
                    0.0,
                )

        elif "Linear" in classname:
            torch.nn.init.xavier_normal_(
                module.weight.data
            )

            if module.bias is not None:
                torch.nn.init.constant_(
                    module.bias.data,
                    0.0,
                )

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(
                0.9,
                0.999,
            ),
            eps=1e-8,
            weight_decay=args.decay_rate,
        )
    else:
        optimizer = torch.optim.SGD(
            classifier.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
        )

    checkpoint_path = (
        checkpoints_dir
        / "best_model.pth"
    )

    (
        start_epoch,
        best_iou,
        resumed,
    ) = load_checkpoint_if_available(
        classifier=classifier,
        optimizer=optimizer,
        checkpoint_path=checkpoint_path,
        log_string=log_string,
    )

    if not resumed:
        classifier = classifier.apply(
            weights_init
        )

    def bn_momentum_adjust(
        module,
        momentum,
    ):
        if isinstance(
            module,
            (
                torch.nn.BatchNorm2d,
                torch.nn.BatchNorm1d,
            ),
        ):
            module.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECAY = 0.5
    MOMENTUM_DECAY_STEP = args.step_size

    global_epoch = start_epoch

    # ------------------------------------------------------------------
    # Training.
    # ------------------------------------------------------------------
    for epoch in range(
        start_epoch,
        args.epoch,
    ):
        log_string(
            "**** Epoch %d (%d/%s) ****"
            % (
                global_epoch + 1,
                epoch + 1,
                args.epoch,
            )
        )

        lr = max(
            args.learning_rate
            * (
                args.lr_decay
                ** (
                    epoch
                    // args.step_size
                )
            ),
            LEARNING_RATE_CLIP,
        )

        log_string(
            "Learning rate:%f"
            % lr
        )

        for param_group in optimizer.param_groups:
            param_group[
                "lr"
            ] = lr

        momentum = (
            MOMENTUM_ORIGINAL
            * (
                MOMENTUM_DECAY
                ** (
                    epoch
                    // MOMENTUM_DECAY_STEP
                )
            )
        )

        momentum = max(
            momentum,
            0.01,
        )

        log_string(
            "BN momentum updated to: %f"
            % momentum
        )

        classifier = classifier.apply(
            lambda module: bn_momentum_adjust(
                module,
                momentum,
            )
        )

        num_batches = len(
            train_loader
        )

        if num_batches == 0:
            raise RuntimeError(
                "Training DataLoader has zero batches. "
                "Reduce batch_size or verify the training dataset."
            )

        total_correct = 0
        total_seen = 0
        loss_sum = 0.0

        classifier = classifier.train()

        for (
            batch_index,
            (
                points,
                target,
            ),
        ) in tqdm(
            enumerate(
                train_loader
            ),
            total=len(
                train_loader
            ),
            smoothing=0.9,
        ):
            optimizer.zero_grad()

            points_np = (
                points.numpy()
            )

            # Keep the original PointNet++ augmentation behavior.
            points_np[
                :,
                :,
                :3,
            ] = provider.rotate_point_cloud_z(
                points_np[
                    :,
                    :,
                    :3,
                ]
            )

            points = torch.from_numpy(
                points_np
            )

            points = points.float().cuda(
                non_blocking=True
            )

            target = target.long().cuda(
                non_blocking=True
            )

            unique_target = torch.unique(
                target
            )

            if torch.any(
                (
                    unique_target < 0
                )
                | (
                    unique_target >= NUM_CLASSES
                )
            ):
                raise RuntimeError(
                    f"Training batch contains invalid labels: "
                    f"{unique_target.detach().cpu().tolist()}"
                )

            points = points.transpose(
                2,
                1,
            )

            seg_pred, trans_feat = classifier(
                points
            )

            seg_pred = (
                seg_pred
                .contiguous()
                .view(
                    -1,
                    NUM_CLASSES,
                )
            )

            batch_label = (
                target
                .view(
                    -1
                )
                .detach()
                .cpu()
                .numpy()
            )

            target_flat = target.view(
                -1
            )

            loss = criterion(
                seg_pred,
                target_flat,
                trans_feat,
                weights,
            )

            kan_reg_loss = compute_kan_regularization(
                classifier,
                lambda_reg=1e-4,
            )

            # For standard PointNet++ this term is zero.
            loss = (
                loss
                + 0.05
                * kan_reg_loss
            )

            loss.backward()
            optimizer.step()

            pred_choice = (
                seg_pred
                .detach()
                .cpu()
                .max(
                    1
                )[1]
                .numpy()
            )

            total_correct += int(
                np.sum(
                    pred_choice
                    == batch_label
                )
            )

            total_seen += int(
                batch_label.size
            )

            loss_sum += float(
                loss.detach().cpu()
            )

        log_string(
            "Training mean loss: %f"
            % (
                loss_sum
                / num_batches
            )
        )

        log_string(
            "Training accuracy: %f"
            % (
                total_correct
                / float(
                    total_seen
                )
            )
        )

        if epoch % 5 == 0:
            savepath = (
                checkpoints_dir
                / "model.pth"
            )

            log_string(
                f"Saving model at {savepath}"
            )

            state = {
                "epoch": epoch,
                "class_avg_iou": best_iou,
                "model_state_dict": classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }

            torch.save(
                state,
                savepath,
            )

        # --------------------------------------------------------------
        # Validation on sampled blocks.
        # --------------------------------------------------------------
        with torch.no_grad():
            num_val_batches = len(
                val_loader
            )

            if num_val_batches == 0:
                raise RuntimeError(
                    "Validation DataLoader has zero batches."
                )

            total_correct = 0
            total_seen = 0
            loss_sum = 0.0

            total_seen_class = np.zeros(
                NUM_CLASSES,
                dtype=np.int64,
            )

            total_correct_class = np.zeros(
                NUM_CLASSES,
                dtype=np.int64,
            )

            total_iou_deno_class = np.zeros(
                NUM_CLASSES,
                dtype=np.int64,
            )

            classifier = classifier.eval()

            log_string(
                "---- EPOCH %03d EVALUATION ----"
                % (
                    global_epoch
                    + 1
                )
            )

            for (
                batch_index,
                (
                    points,
                    target,
                ),
            ) in tqdm(
                enumerate(
                    val_loader
                ),
                total=len(
                    val_loader
                ),
                smoothing=0.9,
            ):
                points = points.float().cuda(
                    non_blocking=True
                )

                target = target.long().cuda(
                    non_blocking=True
                )

                unique_target = torch.unique(
                    target
                )

                if torch.any(
                    (
                        unique_target < 0
                    )
                    | (
                        unique_target >= NUM_CLASSES
                    )
                ):
                    raise RuntimeError(
                        f"Validation batch contains invalid labels: "
                        f"{unique_target.detach().cpu().tolist()}"
                    )

                points = points.transpose(
                    2,
                    1,
                )

                seg_pred, trans_feat = classifier(
                    points
                )

                pred_val = (
                    seg_pred
                    .detach()
                    .cpu()
                    .numpy()
                )

                seg_pred_flat = (
                    seg_pred
                    .contiguous()
                    .view(
                        -1,
                        NUM_CLASSES,
                    )
                )

                batch_label = (
                    target
                    .detach()
                    .cpu()
                    .numpy()
                )

                target_flat = target.view(
                    -1
                )

                loss = criterion(
                    seg_pred_flat,
                    target_flat,
                    trans_feat,
                    weights,
                )

                kan_reg_loss = compute_kan_regularization(
                    classifier,
                    lambda_reg=1e-4,
                )

                # Keep regularization consistent with training.
                loss = (
                    loss
                    + 0.05
                    * kan_reg_loss
                )

                loss_sum += float(
                    loss.detach().cpu()
                )

                pred_val = np.argmax(
                    pred_val,
                    axis=2,
                )

                total_correct += int(
                    np.sum(
                        pred_val
                        == batch_label
                    )
                )

                total_seen += int(
                    batch_label.size
                )

                for class_id in range(
                    NUM_CLASSES
                ):
                    gt_mask = (
                        batch_label
                        == class_id
                    )

                    pred_mask = (
                        pred_val
                        == class_id
                    )

                    total_seen_class[
                        class_id
                    ] += int(
                        np.sum(
                            gt_mask
                        )
                    )

                    total_correct_class[
                        class_id
                    ] += int(
                        np.sum(
                            pred_mask
                            & gt_mask
                        )
                    )

                    total_iou_deno_class[
                        class_id
                    ] += int(
                        np.sum(
                            pred_mask
                            | gt_mask
                        )
                    )

            (
                iou_per_class,
                acc_per_class,
                miou_5class,
                miou_present,
                macc_present,
            ) = compute_validation_metrics(
                total_correct_class,
                total_seen_class,
                total_iou_deno_class,
            )

            class_distribution = (
                total_seen_class.astype(
                    np.float64
                )
                / max(
                    total_seen_class.sum(),
                    1,
                )
            )

            val_oa = (
                total_correct
                / float(
                    max(
                        total_seen,
                        1,
                    )
                )
            )

            log_string(
                "eval mean loss: %f"
                % (
                    loss_sum
                    / num_val_batches
                )
            )

            log_string(
                "eval point avg class IoU (5-class): %f"
                % miou_5class
            )

            log_string(
                "eval point avg class IoU (GT-present): %f"
                % miou_present
            )

            log_string(
                "eval point accuracy: %f"
                % val_oa
            )

            log_string(
                "eval point avg class acc (GT-present): %f"
                % macc_present
            )

            iou_per_class_str = (
                "------- IoU / VAL DISTRIBUTION --------\n"
            )

            for class_id in range(
                NUM_CLASSES
            ):
                iou_per_class_str += (
                    "class %-7s "
                    "GT-freq: %.6f, "
                    "IoU: %.6f, "
                    "Acc: %.6f, "
                    "GT: %d, "
                    "Union: %d\n"
                    % (
                        SEG_LABEL_TO_CAT[
                            class_id
                        ],
                        class_distribution[
                            class_id
                        ],
                        iou_per_class[
                            class_id
                        ],
                        acc_per_class[
                            class_id
                        ],
                        total_seen_class[
                            class_id
                        ],
                        total_iou_deno_class[
                            class_id
                        ],
                    )
                )

            log_string(
                iou_per_class_str
            )

            # Keep checkpoint selection compatible with the original fixed
            # 5-class HOPS metric.
            if miou_5class >= best_iou:
                best_iou = miou_5class

                savepath = (
                    checkpoints_dir
                    / "best_model.pth"
                )

                log_string(
                    f"Saving best model at {savepath}"
                )

                state = {
                    "epoch": epoch,
                    "class_avg_iou": miou_5class,
                    "class_avg_iou_present": miou_present,
                    "model_state_dict": classifier.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }

                torch.save(
                    state,
                    savepath,
                )

            log_string(
                "Best mIoU: %f"
                % best_iou
            )

        global_epoch += 1


if __name__ == "__main__":
    main(
        parse_args()
    )
