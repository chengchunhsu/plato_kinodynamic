

import torch
from plato.policy.transformer_modules import TransformerDecoder, SinusoidalPositionEncoding
from plato.policy.policy_head import GMMHead
import robomimic.utils.tensor_utils as TensorUtils
import torch.nn as nn
from einops import rearrange
from easydict import EasyDict

def safe_device(x, device="cuda"):
    if device == "cpu":
        return x.cpu()
    elif "cuda" in device:
        if torch.cuda.is_available():
            return x.to(device)
        else:
            return x.cpu()

class DynamicsModel(nn.Module):
    def __init__(self):
        super().__init__()
        embed_size = 16

        state_dim = 6
        action_dim = 3

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, embed_size)
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, embed_size)
        )

        output_size = state_dim

        policy_cfg = EasyDict(dict(
            transformer_num_layers=8,
            transformer_num_heads=8,
            transformer_head_output_size=32,
            transformer_mlp_hidden_size=128,
            transformer_dropout=0.1,
        ))

        policy_head_cfg = EasyDict(dict(
            hidden_size=1024,
            num_layers=2,
            min_std=0.0001,
            num_modes=5,
            low_eval_noise=False,
            activation="softplus",
        ))

        self.temporal_transformer = TransformerDecoder(
                    input_size=embed_size,
                    num_layers=policy_cfg.transformer_num_layers,
                    num_heads=policy_cfg.transformer_num_heads,
                    head_output_size=policy_cfg.transformer_head_output_size,
                    mlp_hidden_size=policy_cfg.transformer_mlp_hidden_size,
                    dropout=policy_cfg.transformer_dropout,
                )
        self.gmm_head = GMMHead(
                input_size=embed_size,
                output_size=output_size,
                hidden_size=policy_head_cfg.hidden_size,
                num_layers=policy_head_cfg.num_layers,
                min_std=policy_head_cfg.min_std,
                num_modes=policy_head_cfg.num_modes,
                low_eval_noise=policy_head_cfg.low_eval_noise,
                activation=policy_head_cfg.activation,
        )
        self.temporal_position_encoding_fn = SinusoidalPositionEncoding(embed_size)

        self.device = "cuda"
        

    def temporal_encode(self, x, a=None):
        x = self.state_encoder(x)
        a = self.action_encoder(a)
        x = torch.cat([x, a], dim=2)
        pos_emb = self.temporal_position_encoding_fn(x)
        # print(x.shape, pos_emb.shape)
        x = x + pos_emb.unsqueeze(1)  # (B, T, num_modality, E)
        sh = x.shape
        self.temporal_transformer.compute_mask(x.shape)

        x = rearrange(x, 'b t n e -> b (t n) e')
        x = self.temporal_transformer(x)
        x = x.reshape(*sh)
        return x[:, :, 0]  # (B, T, E)

    def forward(self, x, a=None):
        self.train()        
        x = self.temporal_encode(x, a)
        return self.gmm_head(x)
    
    def predict(self, data):
        data = self.preprocess_input(data)
        x = data["x_seq"]
        a = data["a_seq"]
        self.eval()
        with torch.no_grad():
            x = self.temporal_encode(x, a)
            dist = self.gmm_head(x[:, -1])
            action = dist.sample().detach().cpu()
        return action
    
    def use_low_noise(self):
        self.gmm_head.low_eval_noise = True
    
    def compute_loss(self, data):
        data = self.preprocess_input(data)
        x = data["x_seq"]
        y = data["y_seq"]
        a = data["a_seq"]
        predicted_dist = self.forward(x, a)
        loss = self.gmm_head.loss_fn(predicted_dist, y)
        return loss

    def map_tensor_to_device(self, data):
        """Move data to the device specified by self.cfg.device."""
        return TensorUtils.map_tensor(
            data, lambda x: safe_device(x, device=self.device)
        )
    
    def preprocess_input(self, data):
        data["x_seq"] = self.map_tensor_to_device(data["x_seq"])
        data["a_seq"] = self.map_tensor_to_device(data["a_seq"])
        # augment dimension for subsequent policy inference
        data["x_seq"] = data["x_seq"].unsqueeze(2)
        data["a_seq"] = data["a_seq"]

        if "y_seq" in data:
            data["y_seq"] = self.map_tensor_to_device(data["y_seq"])
            data["y_seq"] = data["y_seq"]
        return data
    
class DynamicsDeltaOutputModel(DynamicsModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
     
    def predict(self, data):
        data = self.preprocess_input(data)
        x = data["x_seq"]
        a = data["a_seq"]
        self.eval()
        with torch.no_grad():
            x = self.temporal_encode(x, a)
            dist = self.gmm_head(x[:, -1])
            action = dist.sample().detach().cpu()
        next_x = data["x_seq"][:, -1].detach().cpu() + action
        return next_x
    
    def use_low_noise(self):
        self.gmm_head.low_eval_noise = True
    
    def compute_loss(self, data):
        data = self.preprocess_input(data)
        x = data["x_seq"]
        y = data["y_seq"]
        a = data["a_seq"]
        predicted_dist = self.forward(x, a)
        loss = self.gmm_head.loss_fn(predicted_dist, y-x[:, :, None, :])
        return loss