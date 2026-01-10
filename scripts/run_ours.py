from itertools import product
import os
from pathlib import Path
import pty
import select
import subprocess

# MipNeRF360 配置参数, copy from train_base and train_big.sh
mipnerf360_params = {
    "base": {
        "bicycle": {"dense": None, "highfeature_lr": None, "grad_abs_thresh": 0.0012, "loss_thresh": None},
        "flowers": {"dense": 0.005, "highfeature_lr": None, "grad_abs_thresh": 0.0015, "loss_thresh": None},
        "garden": {"dense": None, "highfeature_lr": 0.02, "grad_abs_thresh": 0.0008, "loss_thresh": 0.06},
        "stump": {"dense": 0.004, "highfeature_lr": None, "grad_abs_thresh": 0.0015, "loss_thresh": None},
        "treehill": {"dense": 0.01, "highfeature_lr": None, "grad_abs_thresh": 0.002, "loss_thresh": None},
        "room": {"dense": None, "highfeature_lr": 0.02, "grad_abs_thresh": 0.0008, "loss_thresh": None},
        "counter": {"dense": None, "highfeature_lr": 0.02, "grad_abs_thresh": 0.0008, "loss_thresh": None},
        "kitchen": {"dense": None, "highfeature_lr": 0.02, "grad_abs_thresh": 0.0006, "loss_thresh": None},
        "bonsai": {"dense": None, "highfeature_lr": 0.02, "grad_abs_thresh": 0.0006, "loss_thresh": None}
    },
    "big": {
        "bicycle": {"dense": None, "highfeature_lr": None, "grad_abs_thresh": 0.0008, "loss_thresh": None},
        "flowers": {"dense": 0.005, "highfeature_lr": None, "grad_abs_thresh": 0.001, "loss_thresh": None},
        "garden": {"dense": None, "highfeature_lr": 0.02, "grad_abs_thresh": 0.0003, "loss_thresh": 0.06},
        "stump": {"dense": 0.004, "highfeature_lr": None, "grad_abs_thresh": 0.001, "loss_thresh": None},
        "treehill": {"dense": 0.01, "highfeature_lr": None, "grad_abs_thresh": 0.0018, "loss_thresh": None},
        "room": {"dense": None, "highfeature_lr": 0.02, "grad_abs_thresh": 0.0004, "loss_thresh": None},
        "counter": {"dense": None, "highfeature_lr": 0.02, "grad_abs_thresh": 0.0004, "loss_thresh": None},
        "kitchen": {"dense": None, "highfeature_lr": 0.02, "grad_abs_thresh": 0.0002, "loss_thresh": None},
        "bonsai": {"dense": None, "highfeature_lr": 0.02, "grad_abs_thresh": 0.0002, "loss_thresh": None}
    }
}


rogers_params = {
    "base":{
        "tower_0529": {
            "dense": 0.005,
            "highfeature_lr": 0.02,
            "grad_abs_thresh": 0.0012,
            "loss_thresh": 0.06
            }
        }
        ,
    "big":{
        "tower_0529": {
            "dense": 0.005,
            "highfeature_lr": 0.02,
            "grad_abs_thresh": 0.0004,
            "loss_thresh": 0.06
        }
    }
    }

def parmas_to_cmd(params):
    cmd = ""
    for k, v in params.items():
        if v is None:
            continue
        cmd += f"--{k} {v} "
    return cmd

# ============================================================
# FastGS 配置参数
# ============================================================
def get_train_cmd(input: Path, output, image_dir=None, mode="base"):
    scene_name = str(input.parts[-1]).lower()
    # FastGS 特定参数
    if mode == "big":
        densification_interval = 100  # 密集化间隔
    elif mode == "base":
        densification_interval = 500  # 密集化间隔
    
    params = {}
    if "mip" in str(input).lower():
        params_mode = mipnerf360_params[mode]
        for key in params_mode.keys():
            if key in scene_name:
                params = params_mode[key]
    elif "rogers" in str(input).lower():
        params_mode = rogers_params[mode]
        for key in params_mode.keys():
            if key in scene_name:
                params = params_mode[key]

    cmd_ext = parmas_to_cmd(params)
    optimizer_type = "default"    # 优化器类型

    # 路径配置
    image_root = f"/home/matt/cviss/Matt/Dataset/{input}"  # Blendswap/Render/pick/13078_toad
    output_base_dir = f"/home/matt/cviss/Matt/GS-Output"
    output_full_dir = f"{output_base_dir}/FastGS-{mode[0].upper()+mode[1:]}/{output}"  # pick/13078_toad

    # ============================================================
    # FastGS 训练命令
    # ============================================================
    cmd = (
        f"OMP_NUM_THREADS=4 "
        f"CUDA_VISIBLE_DEVICES=0 "
        f"python train.py "
        f"-s {image_root} "
        f"-m {output_full_dir} "
        f"{f'-i {image_dir} ' if image_dir is not None else ''}"
        f"-r 1 "
        f"--eval "
        f"--densification_interval {densification_interval} "
        f"--optimizer_type {optimizer_type} "
        f"--iterations 30000 "
        f"--test_iterations 7000 30000 "
        f"--save_iterations 7000 30000 "
        f"{cmd_ext}"
    )
    # ============================================================
    # FastGS 评估命令
    # ============================================================
    eval_cmd = (
        f"python evaluate_metrics.py -s {image_root} -m {output_full_dir} --iterations 7000 30000"
    )
    
    return cmd, eval_cmd


def run_with_live_output(cmd):
    """运行命令并实时显示输出，正确处理 tqdm"""
    master_fd, slave_fd = pty.openpty()
    
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True
    )
    os.close(slave_fd)
    
    output_lines = []
    
    while True:
        ready, _, _ = select.select([master_fd], [], [], 0.1)
        if ready:
            try:
                data = os.read(master_fd, 1024).decode('utf-8', errors='replace')
                if data:
                    print(data, end='', flush=True)
                    output_lines.append(data)
            except OSError:
                break
        
        if process.poll() is not None:
            # 读取剩余输出
            while True:
                try:
                    data = os.read(master_fd, 1024).decode('utf-8', errors='replace')
                    if not data:
                        break
                    print(data, end='', flush=True)
                    output_lines.append(data)
                except OSError:
                    break
            break
    
    os.close(master_fd)
    return process.returncode, ''.join(output_lines)

# for mode in ["base", "big"]:
#     cmd, eval_cmd = get_train_cmd(input="Blendswap/Render/pick/13078_toad", output="Blendswap/pick/13078_toad", 
#                                   image_dir=None, mode=mode)

#     run_with_live_output(cmd)
#     run_with_live_output(eval_cmd)
#     for factor in [4, 8]:
#         cmd, eval_cmd = get_train_cmd(input="Rogers/Tower_0529", 
#                                       output=f"Rogers/Tower_0529_{factor}", image_dir=f"images_{factor}",
#                                       mode=mode)

#         run_with_live_output(cmd)
#         run_with_live_output(eval_cmd)

dataset_root = Path("/mnt/cviss/Matt/Dataset")
root = dataset_root / "Mip-NeRF360/360_v2/"
for scene_dir in root.iterdir():
    if not scene_dir.is_dir():
        continue
    input_path = Path(f"Mip-NeRF360/360_v2/{scene_dir.name}")
    for factor, mode in product([4, 8], ["base", "big"]):
        output_path = f"Mip-NeRF360/360_v2/{scene_dir.name}_{factor}"
        cmd, eval_cmd = get_train_cmd(input=input_path, output=output_path, image_dir=f"images_{factor}", mode=mode)
        run_with_live_output(cmd)
        run_with_live_output(eval_cmd)
