#!/usr/bin/env python3
"""
Lightweight helper to sweep HNSW parameters.

Configuration can come from repeated --param-set flags or a text file passed with --param-file,
where each non-empty, non-comment line matches the same name=M=..,efc=..,efs=..,k=..,runs=.. format.

For each dataset/parameter-set:
  * Run the main executable N times (with unique output filenames)
  * Parse each run's pruning report
  * Aggregate the query IDs by accumulated pruned-node visits
  * Write the top-K query IDs to a summary file
"""

import argparse
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_dataset(spec: str) -> Tuple[str, Path, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("Dataset spec must follow name=base_path,query_path")
    name, paths = spec.split("=", 1)
    parts = [p.strip() for p in paths.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Dataset spec must provide base and query paths")
    return name.strip(), Path(parts[0]), Path(parts[1])


PARAM_HEADER_RE = re.compile(r"\s*([A-Za-z0-9_\-]+)\s*[:=]\s*(.+)\s*")


def parse_param_set(spec: str) -> Tuple[str, Dict[str, int]]:
    match = PARAM_HEADER_RE.fullmatch(spec)
    if not match:
        raise argparse.ArgumentTypeError("Parameter set must follow name=M=..,efc=..,efs=..,k=..,runs=..")
    name, assignments = match.groups()
    name = name.strip()
    assignments = assignments.strip()
    params: Dict[str, int] = {}
    for chunk in assignments.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise argparse.ArgumentTypeError(f"Invalid assignment '{chunk}' in param set '{name}'")
        key, value = chunk.split("=", 1)
        key = key.strip().lower()
        try:
            params[key] = int(value.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Param '{key}' in set '{name}' must be an integer") from exc

    required_keys = {"m", "efc", "efs", "k", "runs"}
    missing = required_keys - params.keys()
    if missing:
        raise argparse.ArgumentTypeError(
            f"Param set '{name}' missing keys: {', '.join(sorted(missing))}")
    return name.strip(), params


def parse_pruning_file(path: Path) -> Dict[int, int]:
    """Return {query_id: total_pruned_visits} for the file."""
    results: Dict[int, int] = {}
    in_queries = False
    with path.open("r") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[TOP_") and "QUERIES_VISITING_PRUNED_NODES" in line:
                in_queries = True
                continue
            if line.startswith("[TOP_"):
                in_queries = False
                continue
            if in_queries and ":" in line:
                parts = line.split(":")
                if len(parts) >= 2:
                    query_id = int(parts[0])
                    visits = int(parts[1])
                    results[query_id] = visits
    return results


def sanitize(text: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)
    return clean or "unnamed"


def build_command(executable: Path,
                  dataset: Tuple[str, Path, Path],
                  params: Dict[str, int],
                  run_dir: Path,
                  stub: str,
                  top_queries: int) -> Tuple[List[str], Path]:
    dataset_name, base_path, query_path = dataset
    pruning_txt = run_dir / f"{stub}.txt"
    cmd = [
        str(executable),
        "--base",
        str(base_path),
        "--query",
        str(query_path),
        "--M",
        str(params["m"]),
        "--ef-construction",
        str(params["efc"]),
        "--ef-search",
        str(params["efs"]),
        "--k",
        str(params["k"]),
        "--index-out",
        str(run_dir / f"{stub}.bin"),
        "--pruning-out",
        str(pruning_txt),
        "--query-log",
        str(run_dir / f"{stub}_query_log.csv"),
        "--top-queries",
        str(top_queries),
    ]
    return cmd, pruning_txt


def aggregate_runs(run_reports: Iterable[Path], top_queries: int) -> List[Tuple[int, int]]:
    totals: Dict[int, int] = defaultdict(int)
    for report in run_reports:
        per_run = parse_pruning_file(report)
        for query_id, visits in per_run.items():
            totals[query_id] += visits
    sorted_totals = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    return sorted_totals[:top_queries]


def load_param_file(path: Path) -> List[str]:
    specs: List[str] = []
    with path.open("r") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            specs.append(line)
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal HNSW experiment runner.")
    parser.add_argument("--executable", default="./build/main", help="Path to compiled executable")
    parser.add_argument("--dataset", required=True,
                        help="Dataset spec: name=base_path,query_path")
    parser.add_argument("--param-set", action="append",
                        help="Parameter set: name=M=..,efc=..,efs=..,k=..,runs=..")
    parser.add_argument("--param-file", type=Path,
                        help="Path to a text file containing one parameter-set spec per line")
    parser.add_argument("--output-dir", default="experiments_simple", help="Where to store run outputs")
    parser.add_argument("--top-queries", type=int, default=100, help="Top-K queries to keep")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    dataset = parse_dataset(args.dataset)
    param_specs: List[str] = []
    if args.param_file:
        if not args.param_file.exists():
            raise SystemExit(f"Parameter file not found: {args.param_file}")
        param_specs.extend(load_param_file(args.param_file))
    if args.param_set:
        param_specs.extend(args.param_set)
    if not param_specs:
        raise SystemExit("No parameter sets provided. Use --param-set and/or --param-file.")

    param_sets = [parse_param_set(spec) for spec in param_specs]

    exec_path = Path(args.executable)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_label = sanitize(dataset[0])
    base_label = sanitize(dataset[1].stem)
    query_label = sanitize(dataset[2].stem)

    for name, params in param_sets:
        param_label = sanitize(name)
        runs = params["runs"]
        run_dir = output_root / f"{dataset_label}_{param_label}"
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"==> Dataset '{dataset[0]}', param-set '{name}': running {runs} time(s)")
        pruning_reports: List[Path] = []

        for run_idx in range(1, runs + 1):
            stub = f"{dataset_label}_{base_label}_{query_label}_{param_label}_run{run_idx}"
            cmd, pruning_path = build_command(exec_path, dataset, params, run_dir, stub, args.top_queries)

            if args.dry_run:
                print("DRY RUN:", " ".join(cmd))
            else:
                print(f"  Run {run_idx}/{runs} -> {stub}")
                completed = subprocess.run(cmd, capture_output=False, text=True)
                if completed.returncode != 0:
                    print(f"    Run failed with exit code {completed.returncode}, skipping aggregation.")
                    continue

            pruning_reports.append(pruning_path)

        if not pruning_reports:
            print("  No successful runs recorded; skipping aggregation.")
            continue

        if args.dry_run:
            print("  DRY RUN: skipping aggregation.")
            continue

        top_queries = aggregate_runs(pruning_reports, args.top_queries)
        summary_path = run_dir / f"{dataset_label}_{param_label}_top{args.top_queries}.txt"
        with summary_path.open("w") as handle:
            for qid, total in top_queries:
                handle.write(f"{qid}:{total}\n")

        print(f"  Summary written to {summary_path}")


if __name__ == "__main__":
    main()
