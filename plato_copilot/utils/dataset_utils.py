
import h5py
import os
import torch


import numpy as np
from torch.utils.data import Dataset

class PlatoExampleDataset(Dataset):
    def __init__(self, file_path, traj_cap_len=200, seq_len=10):
        self.file_path = file_path
        self.training_data = []  # Placeholder for training data
        self.trajs = []
        self.seq_len = seq_len
        # Load the training data here
        self.load_training_data(traj_cap_len)
        
    def load_training_data(self, traj_cap_len):
        # Load the training data from the file_path
        # Implement your own logic here to load the data
        
        # Example: Loading data from an HDF5 file
        with h5py.File(self.file_path, 'r') as training_data_file:
            data = training_data_file['data']
            for traj_idx, key in enumerate(data.keys()):
                obj_pose = data[key]['object_pose'][()]  # (T,6)
                tool_pose = data[key]['mocap_pose'][()]  # (T,6)
                force_fb = data[key]['force_feedback'][()]  # (T,6)
                actions = data[key]['actions'][()]  # (T,6)

                self.trajs.append({
                    'object_pose': obj_pose,
                    'tool_pose': tool_pose,
                    'force_fb': force_fb,
                    'actions': actions
                })

                T = obj_pose.shape[0]
                cap = min(T - self.seq_len - 1, traj_cap_len)
                for t0 in range(cap):
                    self.training_data.append((traj_idx, t0))
        
    def __len__(self):
        # Return the length of the dataset
        return len(self.training_data)
    
    def __getitem__(self, index):
        traj_idx, t0 = self.training_data[index]
        traj = self.trajs[traj_idx]

        s_obj = traj['object_pose'][t0: t0 + self.seq_len]  # (seq_len,6)
        s_tl = traj['tool_pose'][t0: t0 + self.seq_len]  # (seq_len,6)
        s_fb = traj['force_fb'][t0: t0 + self.seq_len]  # (seq_len,6)
        a_seq = traj['actions'][t0: t0 + self.seq_len]  # (seq_len,6)
        y_seq = traj['object_pose'][t0 + 1: t0 + self.seq_len + 1]  # (seq_len,6)

        x_seq = np.concatenate([s_obj, s_tl, s_fb], axis=1)

        return {
            'x_seq': torch.from_numpy(x_seq).float(),  # (seq_len,18)
            'a_seq': torch.from_numpy(a_seq).float(),  # (seq_len,6)
            'y_seq': torch.from_numpy(y_seq).float()  # (seq_len,6)
        }

class PlatoDeltaDataset(Dataset):
    def __init__(self, file_path, traj_cap_len=70, seq_len=10):
        self.file_path = file_path
        self.training_data = []  # Placeholder for training data
        self.trajs = []
        self.seq_len = seq_len
        # Load the training data here
        self.load_training_data(traj_cap_len)
        
    def load_training_data(self, traj_cap_len):
        # Load the training data from the file_path
        # Implement your own logic here to load the data
        
        # Example: Loading data from an HDF5 file
        with h5py.File(self.file_path, 'r') as training_data_file:
            data = training_data_file['data']
            for count, key in enumerate(data.keys()):
                self.trajs.append((data[key]["object_pose"][()], data[key]["actions"][()]))
                training_data_idx = np.arange(0, traj_cap_len)
                self.training_data += [(count, idx) for idx in training_data_idx]
        
    def __len__(self):
        # Return the length of the dataset
        return len(self.training_data)
    
    def __getitem__(self, index):
        # Return the specific data at the given index
        traj_idx, time_idx = self.training_data[index]
        object_pos_seq = self.trajs[traj_idx][0][time_idx:time_idx+self.seq_len]
        next_object_pos = self.trajs[traj_idx][0][time_idx+1:time_idx+self.seq_len+1]
        delta_object_pos = next_object_pos - object_pos_seq
        actions_seq = self.trajs[traj_idx][1][time_idx:time_idx+self.seq_len]

        object_pos_seq = torch.tensor(object_pos_seq, dtype=torch.float32)
        actions_seq = torch.tensor(actions_seq, dtype=torch.float32)        
        return {
            "x_seq": object_pos_seq,
            "y_seq": delta_object_pos,
            "a_seq": actions_seq
        }