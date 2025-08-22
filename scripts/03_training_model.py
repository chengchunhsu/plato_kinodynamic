import h5py
import os
import numpy as np
os.environ["MUJOCO_GL"] = "egl"

import mujoco
from IPython.display import HTML

import mediapy as media
import init_path
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim

from plato_copilot.utils.misc_utils import create_experiment_folder
from plato_copilot.kinodynamics.dynamics_model import DynamicsDeltaOutputModel, safe_device
from plato_copilot.utils.dataset_utils import PlatoExampleDataset

import argparse

# Parse command line arguments

def main():
    parser = argparse.ArgumentParser(description='Train model on dataset.')
    parser.add_argument('--train_dataset_path', type=str, default="datasets/jenga_sim_data/straight_line/", help='Path to the training dataset')
    parser.add_argument('--val_dataset_path', type=str, default="datasets/jenga_sim_data/straight_line_val/", help='Path to the validation dataset')
    args = parser.parse_args()

    train_dataset_path = args.train_dataset_path
    dataset = PlatoExampleDataset(file_path=f'{train_dataset_path}/training_data.hdf5')

    val_dataset_path = args.val_dataset_path
    val_dataset = PlatoExampleDataset(file_path=f'{val_dataset_path}/training_data.hdf5')

    val_data_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=32)

    dynamics_model = safe_device(DynamicsDeltaOutputModel())

    # Create a data loader with batch size and enable shuffle
    batch_size = 128
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=32)

    adam_optimizer_kwargs = {
        "lr": 1e-3,
    }
    clip_grad_norm = 1.0

    optimizer = optim.Adam(dynamics_model.parameters(), **adam_optimizer_kwargs)

    num_epochs = 60
    print_loss_interval = 5

    best_loss = float("inf")
    experiment_path = create_experiment_folder(f"experiments/{os.path.basename(train_dataset_path)}")

    train_loss_seq = []
    val_loss_seq = []
    for epoch in range(num_epochs):

        training_loss = 0
        for batch in tqdm(data_loader):
            loss = dynamics_model.compute_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            # clip gradient norm
            torch.nn.utils.clip_grad_norm_(dynamics_model.parameters(), clip_grad_norm)
            optimizer.step()
            training_loss += loss.item()

        training_loss /= len(data_loader)
        val_loss = 0

        # Compute validation loss
        with torch.no_grad():
            for val_batch in tqdm(val_data_loader):
                val_loss += dynamics_model.compute_loss(val_batch).item()

        val_loss /= len(val_data_loader)

        # Save the best model based on validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(dynamics_model.state_dict(), f"{experiment_path}/dynamics_model_delta.pth")
            print(f"Best model saved to {experiment_path}/dynamics_model_delta.pth with loss = {best_loss}")

        train_loss_seq.append(training_loss)
        val_loss_seq.append(val_loss)

    torch.save({
        "train_loss_seq": train_loss_seq,
        "val_loss_seq": val_loss_seq
    }, f"{experiment_path}/losses.pth")


if __name__ == "__main__":
    main()