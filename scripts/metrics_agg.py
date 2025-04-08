import sys
import math
from pathlib import Path

import pandas as pd

try:
    from scripts.utils import parse_yaml_file
except ImportError:
    from utils import parse_yaml_file


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        required=True,
        help="Path to object meshes",
    )
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="Number of objects",
    )
    args = parser.parse_args()
    assert args.metrics_dir.exists(), f"[SR] Metrics directory {args.metrics_dir} does not exist"
    metrics_dir = args.metrics_dir

    # ------------------ sr ------------------
    df = {"objcatid": [], "success_rate": []}
    values_acc = 0.0
    count = 0
    for metfile in metrics_dir.glob("*-sr.yml"):
        data = parse_yaml_file(metfile)
        #data["objcatid"] = data["objcatid"].decode("utf-8")
        
        df["objcatid"].append(data["objcatid"])
        df["success_rate"].append(data["success_rate"])
        values_acc += data["success_rate"]
        count += 1
    
    assert count == args.count, f"[EMD] Expected {args.count} objects, found {count}"
        
    df["objcatid"].append("mean")
    df["success_rate"].append(values_acc / count)
    df = pd.DataFrame(df)
    print(df)
    df.to_csv(metrics_dir / "metrics-sr.csv", index=False)
    
    # ------------------ emd ------------------
    df = {"objcatid": [], "emd_vs_mu": [], "emd_vs_std": [],
          "emd_self_mu": [], "emd_self_std": []}
    count = 0
    for metfile in metrics_dir.glob("*-emd.yml"):
        data = parse_yaml_file(metfile)
        df["objcatid"].append(data["objcatid"])
        df["emd_vs_mu"].append(data["emd_vs"][0])
        df["emd_vs_std"].append(data["emd_vs"][1])
        df["emd_self_mu"].append(data["emd_self"][0])
        df["emd_self_std"].append(data["emd_self"][1])
        count += 1
        
    assert count == args.count, f"Expected {args.count} objects, found {count}"
    
    df["objcatid"].append("mean")
    df["emd_vs_mu"].append(sum(df["emd_vs_mu"]) / count)
    df["emd_vs_std"].append(math.sqrt(sum([i**2 for i in df["emd_vs_std"]]) / count))
    df["emd_self_mu"].append(sum(df["emd_self_mu"]) / count)
    df["emd_self_std"].append(math.sqrt(sum([i**2 for i in df["emd_self_std"]]) / count))
    df = pd.DataFrame(df)
    print(df)
    df.to_csv(metrics_dir / "metrics-emd.csv", index=False)
