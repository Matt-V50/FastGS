import json
from pathlib import Path

root = Path("/home/matt/civss/Matt/GS-Output/FastGS")
"Rogers/Tower_0529/"
"""
{
  "count": 86801,
  "iteration": 30000,
  "train": {
    "SSIM": 0.995069146156311,
    "PSNR": 43.402801513671875,
    "LPIPS": 0.00805995799601078
  },
  "test": {
    "SSIM": 0,
    "PSNR": 0,
    "LPIPS": 0
  }
}
"""

# output:
# psnr_7k_train	ssim_7k_train	lpips_7k_train	psnr_7k_test	ssim_7k_test	lpips_7k_test	psnr_30k_train	ssim_30k_train	lpips_30k_train	psnr_30k_test	ssim_30k_test	lpips_30k_test	resolution	gs_number
for scene_dir in root.rglob("results_iter30000.json"):
    results_30k = scene_dir
    results_7k = scene_dir.parent / "results_iter7000.json"
    scene_name = scene_dir.parent.name
    dataset_name = scene_dir.parent.relative_to(root).as_posix()
    
    print(f"Dataset: {dataset_name}, Scene: {scene_name}")
    metrics = {}
    for results in [results_7k, results_30k]:
        with open(results, 'r') as f:
            results = json.load(f)
        for split in ["train", "test"]:
            for metric in ["PSNR", "SSIM", "LPIPS"]:
                key = f"{metric.lower()}_{results['iteration'] // 1000}k_{split}"
                metrics[key] = results[split][metric]
            if split == "train" and results["iteration"] == 30000:        
                metrics["gs_number"] = results["count"]
    for column in ["psnr_7k_train", "ssim_7k_train", "lpips_7k_train",
                   "psnr_7k_test", "ssim_7k_test", "lpips_7k_test",
                   "psnr_30k_train", "ssim_30k_train", "lpips_30k_train",
                   "psnr_30k_test", "ssim_30k_test", "lpips_30k_test",
                   "resolution", "gs_number"]:
        print(metrics.get(column, " "), end="\t")
    print()