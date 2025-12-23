import json
from pathlib import Path

root = Path("/home/matt/cviss/Matt/GS-Output/FastGS-Base")

# output:
# psnr_7k_train	ssim_7k_train	lpips_7k_train	psnr_7k_test	ssim_7k_test	lpips_7k_test	psnr_30k_train	ssim_30k_train	lpips_30k_train	psnr_30k_test	ssim_30k_test	lpips_30k_test	resolution	gs_number
for scene_dir in root.rglob("results_iter30000.json"):
    results_30k = scene_dir
    results_7k = scene_dir.parent / "results_iter7000.json"
    results_timer = scene_dir.parent / "training_time.json"
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
    with open(results_timer, 'r') as f:
        results = json.load(f)
    metrics["train_times"] = results["train_times"]
    metrics["train_render_times"] = results["train_render_times"]
    metrics["train_optimal_times"] = results["train_optimal_times"]
    for column in ["psnr_7k_train", "ssim_7k_train", "lpips_7k_train",
                   "psnr_7k_test", "ssim_7k_test", "lpips_7k_test",
                   "psnr_30k_train", "ssim_30k_train", "lpips_30k_train",
                   "psnr_30k_test", "ssim_30k_test", "lpips_30k_test",
                   "resolution", "gs_number", "train_times", "train_render_times", "train_optimal_times"]:
        print(metrics.get(column, " "), end="\t")
    print()