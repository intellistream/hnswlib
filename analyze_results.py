#!/usr/bin/env python3
"""
Simple script to analyze pruning results
"""

import argparse


def parse_file(filename):
    """Parse the pruning analysis file"""
    pruned_nodes = {}
    queries = []

    with open(filename, "r") as f:
        lines = f.readlines()

    current_section = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("[TOP_") and line.endswith("_PRUNED_NODES]"):
            current_section = "pruned_nodes"
            continue
        if line.startswith("[TOP_") and "QUERIES_VISITING_PRUNED_NODES" in line:
            current_section = "queries"
            continue

        if current_section == "pruned_nodes" and ":" in line:
            parts = line.split(":")
            if len(parts) == 2:
                node_id = int(parts[0])
                count = int(parts[1])
                pruned_nodes[node_id] = count

        elif current_section == "queries" and ":" in line:
            parts = line.split(":")
            if len(parts) >= 3:
                query_id = int(parts[0])
                visited = int(parts[1])
                queries.append((query_id, visited))

    return pruned_nodes, queries


def main():
    parser = argparse.ArgumentParser(description="Analyze HNSW pruning results.")
    parser.add_argument("--input", default="hnsw_pruning_analysis.txt", help="Path to pruning analysis txt file")
    parser.add_argument("--output", default="top_query_idx.txt", help="File to write comma-separated query indices")
    parser.add_argument("--top", type=int, default=100, help="Number of queries to keep in the output list")
    args = parser.parse_args()

    pruned_nodes, queries = parse_file(args.input)

    queries.sort(key=lambda x: x[1], reverse=True)

    top_n = max(0, args.top)
    query_indices = [str(query_id) for query_id, _ in queries[:top_n]]

    with open(args.output, "w") as f:
        f.write(",".join(query_indices))

    print(f"Found {len(queries)} queries")
    for i, (query_id, visited) in enumerate(queries[: min(10, len(queries))]):
        print(f"  {i + 1}. Query {query_id}: {visited} pruned nodes visited")

    print(f"\nResults written to: {args.output}")


if __name__ == "__main__":
    main()
