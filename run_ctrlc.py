import os
import os.path as osp
import argparse
from datetime import date
import json
import random
import time
from pathlib import Path
from typing import List, Any
import numpy as np
import numpy.linalg as LA
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import csv
import submitit

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import datasets
import util.misc as utils
from datasets import build_image_dataset
from models import build_model
from config import cfg

def get_args_parser():
    parser = argparse.ArgumentParser('Set CTRL-C', add_help=False)
    parser.add_argument('--config-file', 
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        default='config-files/ctrl-c.yaml')
    parser.add_argument('--imagedirs', type=Path, default='/checkpoint/cywu/ego4d_subsets_for_sfm/easy9')
    parser.add_argument("--sample_n_imgs", default=10, type=int, help="Number of images to sample from a image directory")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--slurm_partition', default='devlab')
    parser.add_argument("--opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    return parser


def to_device(data, device):
    if type(data) == dict:
        return {k: v.to(device) for k, v in data.items()}
    return [{k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in t.items()} for t in data]

def main(cfg, imagedirs, sample_n_imgs, num_workers, slurm_partition):
    device = torch.device(cfg.DEVICE)
    
    model, _ = build_model(cfg)
    model.to(device)

    checkpoint = torch.load('logs/checkpoint.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.eval()

    img_dirs = [p for p in imagedirs.iterdir() if p.is_dir]

    if num_workers == 0:
        for i, img_dir in enumerate(img_dirs):
            compute(cfg, model, img_dir, sample_n_imgs, device)
    else:
        run_on_slurm(cfg, model, img_dirs, sample_n_imgs, device, num_workers, slurm_partition)

def split_tasks(tasks: List[Any], num_workers: int) -> List[List[Any]]:
    """ Split a list of tasks up into equal-ish sized chunks """
    # Figure out how many tasks each worker should handle. We want to split as
    # evenly as possible, so the difference in number of tasks per worker
    # varies by at most one. We first split evenly (rounding down) then go back
    # and assign one extra task per worker as needed.
    tasks_per_worker = [len(tasks) // num_workers for _ in range(num_workers)]
    for i in range(num_workers):
        if sum(tasks_per_worker) == len(tasks):
            break
        tasks_per_worker[i] += 1
    assert sum(tasks_per_worker) == len(tasks)

    # Now actually split tasks
    sharded_tasks = [[]]
    for task in tasks:
        worker_idx = len(sharded_tasks) - 1
        if len(sharded_tasks[worker_idx]) == tasks_per_worker[worker_idx]:
            sharded_tasks.append([])
        sharded_tasks[-1].append(task)
    for num, shard in zip(tasks_per_worker, sharded_tasks):
        assert num == len(shard)

    return sharded_tasks


def run_on_slurm(cfg, model, img_dirs, sample_n_imgs, device, num_workers, slurm_partition):
    tasks = [(cfg, model, i, sample_n_imgs, device) for i in zip(img_dirs)]
    sharded_tasks = split_tasks(tasks, num_workers)

    slurm_folder = cfg.OUTPUT_DIR + '/slurm'
    executor = submitit.AutoExecutor(folder=slurm_folder)
    executor_params = {
        'gpus_per_node': 1,
        'tasks_per_node': 1,
        'mem_gb': 64,
        'cpus_per_task': 8,
        'nodes': 1,
        'timeout_min': 60 * 48,
        'slurm_partition': slurm_partition,
    }
    executor.update_parameters(**executor_params)
    executor.update_parameters(slurm_srun_args=["-vv", "--cpu-bind", "none"])
    jobs = []
    with executor.batch():
        for shard in sharded_tasks:
            job = executor.submit(TaskRunner(shard))
            jobs.append(job)
    for job in jobs:
        print(f'Submitted job {job.job_id}')
    print(f'Job dir: {slurm_folder}')


class TaskRunner:
    def __init__(self, tasks):
        self.tasks = tasks

    def __call__(self):
        num_tasks = len(self.tasks)
        for i, (cfg, model, img_dir, sample_n_imgs, device) in enumerate(self.tasks):
            print(f'Computing intrinsics {i + 1} / {num_tasks}')
            compute(cfg, model, img_dir[0], sample_n_imgs, device)

def compute(cfg, model, img_dir, sample_n_imgs, device):
    print("Processing: ", img_dir)
    output_file = cfg.OUTPUT_DIR + "/" + img_dir.name + ".npy"
    if os.path.isfile(output_file):
        print("Output file already exists! Skipping.")
        return
    img_paths = [str(p) for p in img_dir.iterdir()]
    if len(img_paths) == 0:
        print("Input folder empty! Skipping.")
        return
    if sample_n_imgs > 0:
        idx = np.round(np.linspace(0, len(img_paths) - 1, sample_n_imgs)).astype(int)
        img_paths = np.array(img_paths)[idx].tolist()
    dataset_test = build_image_dataset(image_set=img_paths, cfg=cfg)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = DataLoader(dataset_test, 1, sampler=sampler_test,
                                drop_last=False, 
                                collate_fn=utils.collate_fn, 
                                num_workers=1)
    
    all_fxs = []
    for i, (samples, extra_samples, targets) in enumerate(tqdm(data_loader_test)):
        with torch.no_grad():
            samples = samples.to(device)
            extra_samples = to_device(extra_samples, device)
            outputs, extra_info = model(samples, extra_samples)

            zvp = outputs['pred_zvp'].to('cpu')[0].numpy()
            fovy = outputs['pred_fovy'].to('cpu')[0].numpy()
            hl = outputs['pred_hl'].to('cpu')[0].numpy()

            img_sz = targets[0]['org_sz']
            crop_sz = targets[0]['crop_sz']
            filename = targets[0]['filename']
            filename = osp.splitext(filename)[0]

            focal = (img_sz[1]/2.0)/np.tan(fovy/2.0)
            all_fxs.append(focal)
    cx = img_sz[1]/2.0
    cy = img_sz[0]/2.0
    fx = np.median(np.array(all_fxs))
    intr = [fx, fx, cx, cy]
    np.save(output_file, intr)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser('CTRL-C script', parents=[get_args_parser()])
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    if cfg.OUTPUT_DIR:
        Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    main(cfg, args.imagedirs, args.sample_n_imgs, args.num_workers, args.slurm_partition)