# scripts/compose_latency_report.py
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional


def _ensure_parent_dir(path: Optional[str]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _read_latest_record(jsonl_path: str, sample_id: str, role: str) -> Optional[Dict[str, Any]]:
    if not jsonl_path or not os.path.isfile(jsonl_path):
        return None
    found = None
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("sample_id") == sample_id and obj.get("role") == role:
                found = obj
    return found


def _append_jsonl(path: Optional[str], obj: Dict[str, Any]) -> None:
    if not path:
        return
    _ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _append_csv(path: Optional[str], row: Dict[str, Any], fieldnames: list) -> None:
    if not path:
        return
    _ensure_parent_dir(path)
    file_exists = os.path.isfile(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow({k: row.get(k, None) for k in fieldnames})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cloud_jsonl", type=str, required=True, help="JSONL file that contains role=cloud records.")
    p.add_argument("--edge_jsonl", type=str, required=True, help="JSONL file that contains role=edge records.")
    p.add_argument("--sample_id", type=str, required=True, help="Which sample_id to compose.")
    p.add_argument("--bandwidth_mbps", type=float, required=True, help="Bandwidth in Mbps (used for hand-calculated comm overhead).")

    p.add_argument("--out_jsonl", type=str, required=True, help="Append report record to this JSONL.")
    p.add_argument("--out_csv", type=str, default=None, help="Append report row to this CSV (optional).")

    args = p.parse_args()

    cloud = _read_latest_record(args.cloud_jsonl, args.sample_id, role="cloud")
    edge = _read_latest_record(args.edge_jsonl, args.sample_id, role="edge")

    if cloud is None:
        raise RuntimeError(f"No cloud record found for sample_id={args.sample_id} in {args.cloud_jsonl}")
    if edge is None:
        raise RuntimeError(f"No edge record found for sample_id={args.sample_id} in {args.edge_jsonl}")

    cloud_lat = float(cloud.get("timing", {}).get("cloud_latency_sec", 0.0))
    edge_lat = float(edge.get("timing", {}).get("edge_latency_sec", 0.0))

    key_bytes = int(cloud.get("io", {}).get("keyframes_file_bytes", 0))
    volume_mb = key_bytes / 1_000_000.0  # decimal MB

    bw_MBps = args.bandwidth_mbps / 8.0
    comm_overhead = (volume_mb / bw_MBps) if bw_MBps > 0 else None

    total = cloud_lat + edge_lat + (comm_overhead if comm_overhead is not None else 0.0)

    report = {
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "role": "report",
        "sample_id": args.sample_id,
        "bandwidth_mbps": float(args.bandwidth_mbps),
        "timing": {
            "cloud_latency_sec": cloud_lat,
            "edge_latency_sec": edge_lat,
            "communication_overhead_sec": comm_overhead,
            "total_latency_sec": total,
        },
        "io": {
            "communication_volume_mb": volume_mb,
            "keyframes_file_bytes": key_bytes,
            "keyframes_video_path": cloud.get("io", {}).get("keyframes_video_path", None),
            "edge_input_keyframes_video": edge.get("io", {}).get("input_keyframes_video", None),
            "edge_output_video": edge.get("io", {}).get("output_video", None),
        },
        "ref": {
            "cloud_ts": cloud.get("ts"),
            "edge_ts": edge.get("ts"),
            "prompt": cloud.get("prompt"),
        },
    }

    _append_jsonl(args.out_jsonl, report)

    if args.out_csv:
        fields = [
            "ts", "role", "sample_id",
            "bandwidth_mbps",
            "cloud_latency_sec", "edge_latency_sec",
            "communication_volume_mb", "communication_overhead_sec",
            "total_latency_sec",
            "keyframes_video_path",
            "edge_input_keyframes_video",
            "edge_output_video",
            "prompt",
        ]
        row = {
            "ts": report["ts"],
            "role": "report",
            "sample_id": report["sample_id"],
            "bandwidth_mbps": report["bandwidth_mbps"],
            "cloud_latency_sec": report["timing"]["cloud_latency_sec"],
            "edge_latency_sec": report["timing"]["edge_latency_sec"],
            "communication_volume_mb": report["io"]["communication_volume_mb"],
            "communication_overhead_sec": report["timing"]["communication_overhead_sec"],
            "total_latency_sec": report["timing"]["total_latency_sec"],
            "keyframes_video_path": report["io"]["keyframes_video_path"],
            "edge_input_keyframes_video": report["io"]["edge_input_keyframes_video"],
            "edge_output_video": report["io"]["edge_output_video"],
            "prompt": report["ref"]["prompt"],
        }
        _append_csv(args.out_csv, row, fields)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
