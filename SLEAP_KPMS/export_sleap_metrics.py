#!/usr/bin/env python3
"""
SLEAP Metrics Exporter
Author: Lucas Patel (lpatel@ucsd.edu)

Extracts metrics from Amanda's random af SLEAP .npz files that for 
whatever reason do not work with the conventional model reporting
functionality of SLEAP and instead exports to structured plaintext.
Works without requiring the sleap module installed. Warning this
is a super bespoke program I made for Amanda on the fly and
might have bugs, so take everything with a grain of salt!

Usage:
    python export_sleap_metrics.py metrics.train.npz -o output.txt
"""

import argparse
import struct
import sys
import zipfile
from pathlib import Path
import numpy as np


def extract_int(data: bytes, key: bytes) -> int | None:
    """Extract int32 value following a key in pickle data."""
    idx = data.find(key)
    if idx == -1:
        return None
    region = data[idx : idx + 100]
    marker = region.find(b"C\x04")
    if marker != -1:
        return struct.unpack("<i", data[idx + marker + 2 : idx + marker + 6])[0]
    return None


def extract_float(data: bytes, key: bytes) -> float | None:
    """Extract float64 value following a key in pickle data."""
    idx = data.find(key)
    if idx == -1:
        return None
    region = data[idx : idx + 100]
    marker = region.find(b"C\x08")
    if marker != -1:
        return struct.unpack("<d", data[idx + marker + 2 : idx + marker + 10])[0]
    return None


def extract_distances(data: bytes) -> np.ndarray:
    """Extract distance error array from pickle data."""
    idx = data.find(b"dist.dists")
    if idx == -1:
        return np.array([])

    region = data[idx : idx + 50000]
    distances = []

    for start in range(1000, min(40000, len(region)), 8):
        try:
            v = struct.unpack("<d", region[start : start + 8])[0]
            if 0 < v < 500:
                distances.append(v)
            elif len(distances) > 50:
                break
        except struct.error:
            break

    return np.array(distances) if distances else np.array([])


def load_npz_raw(filepath: Path) -> bytes:
    """Load raw pickle data from npz file."""
    with zipfile.ZipFile(filepath, "r") as zf:
        names = zf.namelist()
        if not names:
            raise ValueError("Empty npz file")

        with zf.open(names[0]) as f:
            raw = f.read()

    # Skip NPY header
    header_len = int.from_bytes(raw[8:10], "little")
    return raw[10 + header_len :]


def parse_metrics(data: bytes) -> dict:
    """Parse all metrics from raw pickle data."""
    metrics = {}

    # Visibility metrics
    metrics["fp"] = extract_int(data, b"vis.fp")
    metrics["tn"] = extract_int(data, b"vis.tn")
    metrics["fn"] = extract_int(data, b"vis.fn")
    metrics["precision"] = extract_float(data, b"vis.precision")
    metrics["recall"] = extract_float(data, b"vis.recall")

    # Calculate TP from recall if possible
    if metrics["recall"] and metrics["fn"] and metrics["recall"] < 1:
        metrics["tp"] = int(metrics["recall"] * metrics["fn"] / (1 - metrics["recall"]))
    else:
        metrics["tp"] = extract_int(data, b"vis.tp")

    # Calculate F1
    if metrics["precision"] and metrics["recall"]:
        p, r = metrics["precision"], metrics["recall"]
        metrics["f1"] = 2 * p * r / (p + r)
    else:
        metrics["f1"] = None

    # Localization metrics
    metrics["oks_mAP"] = extract_float(data, b"oks.mAP")
    metrics["oks_voc_mAP"] = extract_float(data, b"oks_voc.mAP")
    metrics["pck_mAP"] = extract_float(data, b"pck.mAP")
    metrics["pck_voc_mAP"] = extract_float(data, b"pck_voc.mAP")

    # Distance errors
    metrics["distances"] = extract_distances(data)

    return metrics


def format_value(v, fmt=".6f") -> str:
    """Format a value, handling None."""
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:{fmt}}"
    return str(v)


def generate_report(metrics: dict, source_file: str) -> str:
    """Generate structured plaintext report."""
    lines = [
        "# SLEAP Training Metrics Export",
        f"# Source: {source_file}",
        "",
        "=" * 80,
        "VISIBILITY METRICS (Keypoint Detection)",
        "=" * 80,
        f"true_positives:   {format_value(metrics['tp'])}",
        f"false_positives:  {format_value(metrics['fp'])}",
        f"true_negatives:   {format_value(metrics['tn'])}",
        f"false_negatives:  {format_value(metrics['fn'])}",
        f"precision:        {format_value(metrics['precision'])}",
        f"recall:           {format_value(metrics['recall'])}",
        f"f1_score:         {format_value(metrics['f1'])}",
        "",
        "=" * 80,
        "LOCALIZATION METRICS (Keypoint Accuracy)",
        "=" * 80,
        f"oks_mAP:          {format_value(metrics['oks_mAP'])}",
        f"oks_voc_mAP:      {format_value(metrics['oks_voc_mAP'])}",
        f"pck_mAP:          {format_value(metrics['pck_mAP'])}",
        f"pck_voc_mAP:      {format_value(metrics['pck_voc_mAP'])}",
    ]

    dist = metrics["distances"]
    if len(dist) > 0:
        lines.extend([
            "",
            "=" * 80,
            "DISTANCE ERROR STATISTICS (pixels)",
            "=" * 80,
            f"n_samples:        {len(dist)}",
            f"mean:             {np.mean(dist):.4f}",
            f"std:              {np.std(dist):.4f}",
            f"min:              {np.min(dist):.4f}",
            f"max:              {np.max(dist):.4f}",
            f"p25:              {np.percentile(dist, 25):.4f}",
            f"p50_median:       {np.percentile(dist, 50):.4f}",
            f"p75:              {np.percentile(dist, 75):.4f}",
            f"p90:              {np.percentile(dist, 90):.4f}",
            f"p95:              {np.percentile(dist, 95):.4f}",
            f"p99:              {np.percentile(dist, 99):.4f}",
            "",
            "=" * 80,
            "DISTANCE ERROR VALUES (all samples, pixels)",
            "=" * 80,
        ])
        lines.extend([f"{d:.4f}" for d in dist])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Export SLEAP metrics .npz to structured plaintext"
    )
    parser.add_argument("input", help="Input .npz file")
    parser.add_argument("-o", "--output", help="Output .txt file (default: input.txt)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.with_suffix(".txt")

    print(f"Loading {input_path}...")
    data = load_npz_raw(input_path)

    print("Parsing metrics...")
    metrics = parse_metrics(data)

    report = generate_report(metrics, input_path.name)

    output_path.write_text(report)
    print(f"Exported to {output_path}")

    # Print summary
    print("\nSummary:")
    print(f"  Precision: {format_value(metrics['precision'])}")
    print(f"  Recall:    {format_value(metrics['recall'])}")
    print(f"  F1 Score:  {format_value(metrics['f1'])}")
    if metrics["oks_voc_mAP"]:
        print(f"  OKS mAP:   {format_value(metrics['oks_voc_mAP'])}")
    if len(metrics["distances"]) > 0:
        print(f"  Median error: {np.median(metrics['distances']):.2f} px")


if __name__ == "__main__":
    main()
