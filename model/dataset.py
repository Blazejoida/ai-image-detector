from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import random
import os
from config import (
    DATA_DIR, BATCH_SIZE, IMAGE_SIZE,
    NORMALIZE_MEAN, NORMALIZE_STD
)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}
FALLBACK_TEST_RATIO = 0.2
FALLBACK_SPLIT_SEED = 42

def get_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ])
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])


def _count_valid_images(path: str) -> int:
    if not os.path.isdir(path):
        return 0
    return sum(
        1
        for root, _, files in os.walk(path)
        for file_name in files
        if os.path.splitext(file_name)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def _split_indices_by_class(samples, test_ratio: float, seed: int):
    by_class = defaultdict(list)
    for sample_idx, (_, class_idx) in enumerate(samples):
        by_class[class_idx].append(sample_idx)

    rng = random.Random(seed)
    train_indices = []
    test_indices = []

    for indices in by_class.values():
        if len(indices) < 2:
            raise ValueError(
                "Each class must have at least 2 images to create a train/test split."
            )

        shuffled = indices[:]
        rng.shuffle(shuffled)

        test_count = max(1, int(round(len(shuffled) * test_ratio)))
        test_count = min(test_count, len(shuffled) - 1)

        test_indices.extend(shuffled[:test_count])
        train_indices.extend(shuffled[test_count:])

    return train_indices, test_indices


def get_loaders():
    data_root = os.path.abspath(DATA_DIR)
    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")

    train_full = datasets.ImageFolder(
        train_dir,
        transform=get_transforms(train=True)
    )

    if len(train_full) == 0:
        raise FileNotFoundError(
            f"No supported images found in training directory: {train_dir}. "
            f"Expected class folders with images (e.g., train/fake, train/real)."
        )

    test_fake_count = _count_valid_images(os.path.join(test_dir, "fake"))
    test_real_count = _count_valid_images(os.path.join(test_dir, "real"))
    use_test_split = test_fake_count > 0 and test_real_count > 0

    if use_test_split:
        train_dataset = train_full
        test_dataset = datasets.ImageFolder(
            test_dir,
            transform=get_transforms(train=False)
        )
    else:
        print(
            "Warning: data/test is empty or incomplete. "
            "Using a reproducible split from data/train instead."
        )
        eval_full = datasets.ImageFolder(
            train_dir,
            transform=get_transforms(train=False)
        )
        train_indices, test_indices = _split_indices_by_class(
            train_full.samples,
            test_ratio=FALLBACK_TEST_RATIO,
            seed=FALLBACK_SPLIT_SEED,
        )
        train_dataset = Subset(train_full, train_indices)
        test_dataset = Subset(eval_full, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Data dir : {data_root}")
    print(f"Classes  : {train_full.class_to_idx}")
    print(f"Train    : {len(train_dataset)} samples")
    print(f"Test     : {len(test_dataset)} samples")

    return train_loader, test_loader, train_full.classes
