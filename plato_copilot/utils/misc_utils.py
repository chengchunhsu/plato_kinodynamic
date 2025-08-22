import os
import torch
import json
import numpy as np
import random

from pathlib import Path

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def torch_save_model(policy, model_path, cfg=None):
    torch.save({"state_dict": policy.state_dict(), "cfg": cfg}, model_path)
    with open(model_path.replace(".pth", ".json"), "w") as f:
        json.dump(cfg, f, cls=NpEncoder, indent=4)

def torch_load_model(model_path):
    model_dict = torch.load(model_path)
    cfg = None
    if "cfg" in model_dict:
        cfg = model_dict["cfg"]
    return model_dict["state_dict"], cfg

def save_run_cfg(output_dir, cfg):
    with open(os.path.join(output_dir, "cfg.json"), "w") as f:
        json.dump(cfg, f, cls=NpEncoder, indent=4)

def load_run_cfg(output_dir, config_name="cfg"):
    with open(os.path.join(output_dir, f"{config_name}.json"), "r") as f:
        return json.load(f)

def set_manual_seeds(seed: int, deterministic: bool = False):
    if seed is not None:
        assert(type(seed) is int)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)            
        print(f"Setting random seeds: {seed}")
    else:
        print(f"Not setting random seeds. The experiment might not be reproducible.")

def create_experiment_folder(output_dir, run_prefix="run", debug=False):
    experiment_id = 0
    for path in Path(output_dir).glob(f'{run_prefix}_*'):
        if not path.is_dir():
            continue
        try:
            folder_id = int(str(path).split(f'{run_prefix}_')[-1])
            if folder_id > experiment_id:
                experiment_id = folder_id
        except BaseException:
            pass
    experiment_id += 1
    if debug:
        experiment_id = 0
    output_dir += f"/{run_prefix}_{experiment_id:03d}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir