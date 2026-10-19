"""Summarise the metrics of every run under a results directory into tables.

    python scripts/collect_metrics.py outputs/

Each run's evaluation/metrics_<iteration>.json is one row, and each dataset gets a mean row;
the tables are printed and written as CSV into the results directory.
"""
import argparse
import csv
import json
import os
import sys

# section -> [(key, label, decimals)], and the CSV it is written to
FAMILIES = [
    ("xray", [
        ("3d/psnr_3d", "PSNR_3D", 2),
        ("3d/ssim_3d", "SSIM_3D", 4),
        ("2d/test/psnr", "PSNR_2D", 2),
        ("2d/test/ssim", "SSIM_2D", 4),
    ], "xray_metrics_summary.csv"),
    ("rgb", [
        ("psnr", "PSNR", 2),
        ("ssim", "SSIM", 4),
        ("lpips", "LPIPS", 4),
    ], "rgb_metrics_summary.csv"),
    ("model", [
        ("N_total", "N_total", 0),
        ("N_surface", "N_surface", 0),
        ("storage_MB", "Storage_MB", 2),
    ], "param_stats_summary.csv"),
]


def read_metrics_file(evaluation_dir):
    """The metrics of the latest iteration under `evaluation_dir`."""
    iterations = [int(name[len("metrics_"):-len(".json")]) for name in os.listdir(evaluation_dir)
                  if name.startswith("metrics_") and name.endswith(".json")]
    if not iterations:
        raise FileNotFoundError(f"no metrics_*.json under {evaluation_dir}")
    with open(os.path.join(evaluation_dir, f"metrics_{max(iterations)}.json")) as fh:
        return json.load(fh)


def iter_scenes(results_dir):
    """(label, path) for every run (a directory holding evaluation/) below `results_dir`."""
    results_dir = os.path.abspath(results_dir)
    for dirpath, dirnames, _ in os.walk(results_dir):
        if os.path.isdir(os.path.join(dirpath, "evaluation")):
            label = os.path.relpath(dirpath, results_dir)
            yield (os.path.basename(dirpath) if label == "." else label), dirpath
            dirnames[:] = []


def collect(results_dir, section, metrics):
    """{scene: {label: value}} for one metric family, skipping scenes that lack it."""
    out = {}
    for scene, scene_dir in iter_scenes(results_dir):
        try:
            data = read_metrics_file(os.path.join(scene_dir, "evaluation"))
        except (OSError, ValueError) as exc:
            print(f"warning: no metrics for {scene}: {exc}", file=sys.stderr)
            continue
        values = data.get(section, {})
        row = {label: values[key] for key, label, _ in metrics if values.get(key) is not None}
        if row:
            out[scene] = row
    return out


def mean_over_scenes(rows, label):
    values = [r[label] for r in rows.values() if label in r]
    return sum(values) / len(values) if values else None


def format_cell(value, decimals):
    return "" if value is None else f"{value:.{decimals}f}"


def print_table(title, rows, metrics, csv_path):
    if not rows:
        print(f"\n{title}: nothing found")
        return
    labels = [label for _, label, _ in metrics]
    decimals = {label: d for _, label, d in metrics}
    # the scenes of each dataset (the first part of "dataset/scene"), then their mean
    datasets = {}
    for scene in sorted(rows):
        datasets.setdefault(scene.split("/")[0] if "/" in scene else "", []).append(scene)
    table = []
    for dataset, scenes in datasets.items():
        table += [(scene, rows[scene]) for scene in scenes]
        table.append((f"{dataset} mean" if dataset else "Mean",
                      {l: mean_over_scenes({s: rows[s] for s in scenes}, l) for l in labels}))
    width = max(len(name) for name, _ in table)

    print(f"\n{title}")
    print("  " + "Scene".ljust(width) + "".join(f"{lab:>16}" for lab in labels))
    for name, row in table:
        print("  " + name.ljust(width) + "".join(f"{format_cell(row.get(l), decimals[l]):>16}" for l in labels))

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Scene"] + labels)
        for name, row in table:
            writer.writerow([name] + ["" if row.get(l) is None else row[l] for l in labels])
    print(f"  -> {csv_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", help="directory of run directories, one per scene")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        parser.error(f"{args.results_dir} is not a directory")
    print(f"Collecting metrics from: {args.results_dir}")
    for section, metrics, csv_name in FAMILIES:
        print_table(section, collect(args.results_dir, section, metrics), metrics, os.path.join(args.results_dir, csv_name))


if __name__ == "__main__":
    main()
