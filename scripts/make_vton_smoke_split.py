import argparse
import os
import random


def read_pairs(path):
    with open(path, "r", encoding="utf-8") as handle:
        pairs = [tuple(line.split()[:2]) for line in handle if len(line.split()) >= 2]
    if not pairs:
        raise ValueError(f"No pairs found in {path}")
    return pairs


def existing_people(dataset_root, split, pairs):
    image_dir = os.path.join(dataset_root, split, "image")
    cloth_dir = os.path.join(dataset_root, split, "cloth")
    mask_dir = os.path.join(dataset_root, split, "agnostic-mask")
    output = []
    for person, _ in pairs:
        stem = os.path.splitext(person)[0]
        mask_candidates = (
            os.path.join(mask_dir, person),
            os.path.join(mask_dir, f"{stem}_mask.png"),
            os.path.join(mask_dir, f"{stem}.png"),
        )
        if (
            os.path.isfile(os.path.join(image_dir, person))
            and os.path.isfile(os.path.join(cloth_dir, person))
            and any(os.path.isfile(path) for path in mask_candidates)
        ):
            output.append(person)
    return list(dict.fromkeys(output))


def write_pairs(path, pairs):
    with open(path, "w", encoding="utf-8") as handle:
        for person, garment in pairs:
            handle.write(f"{person} {garment}\n")


def main(args):
    dataset_root = os.path.abspath(args.dataset_root)
    output_dir = os.path.abspath(args.output_dir or os.path.join(dataset_root, "smoke32"))
    train_people = existing_people(
        dataset_root,
        "train",
        read_pairs(os.path.join(dataset_root, "train_pairs.txt")),
    )
    test_people = existing_people(
        dataset_root,
        "test",
        read_pairs(os.path.join(dataset_root, "test_pairs.txt")),
    )
    if len(train_people) < args.train_count:
        raise ValueError(f"Need {args.train_count} complete training samples, found {len(train_people)}")
    if len(test_people) < 2:
        raise ValueError("Need at least two complete test samples for paired and unpaired validation")

    random.Random(args.seed).shuffle(train_people)
    random.Random(args.seed + 1).shuffle(test_people)
    train_pairs = [(name, name) for name in train_people[: args.train_count]]
    validation_person = test_people[0]
    unpaired_garment = next(name for name in test_people[1:] if name != validation_person)
    validation_pairs = [
        (validation_person, validation_person),
        (validation_person, unpaired_garment),
    ]

    os.makedirs(output_dir, exist_ok=True)
    write_pairs(os.path.join(output_dir, "train_pairs_32.txt"), train_pairs)
    write_pairs(os.path.join(output_dir, "validation_pair_unpair.txt"), validation_pairs)
    print(f"Wrote {len(train_pairs)} training pairs to {output_dir}")
    print(f"Validation person: {validation_person}")
    print(f"Paired garment: {validation_person}")
    print(f"Unpaired garment: {unpaired_garment}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2025)
    main(parser.parse_args())

