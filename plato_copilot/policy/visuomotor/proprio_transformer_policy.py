import torch
from plato_copilot.neural_networks.transformer_modules import TransformerDecoder, SinusoidalPositionEncoding
from plato_copilot.neural_networks.policy_head import GMMHead
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

class ProprioTransformerPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        token_size = cfg.token_size

        self.obs_shape_dict = cfg.obs_shape_dict


        self.modality_encoders = nn.ModuleDict()
        for modality, input_dim in self.obs_shape_dict.items():
            print(modality, input_dim)
            self.modality_encoders[modality] = nn.Sequential(
                nn.Linear(input_dim[0], token_size)
        )


        output_size = cfg.ac_dim

        gmm_head_cfg = cfg.gmm_head_cfg

        self.temporal_transformer = TransformerDecoder(
                    input_size=token_size,
                    num_layers=cfg.transformer_cfg.transformer_num_layers,
                    num_heads=cfg.transformer_cfg.transformer_num_heads,
                    head_output_size=cfg.transformer_cfg.transformer_head_output_size,
                    mlp_hidden_size=cfg.transformer_cfg.transformer_mlp_hidden_size,
                    dropout=cfg.transformer_cfg.transformer_dropout,
                )
        self.gmm_head = GMMHead(
                input_size=token_size,
                output_size=output_size,
                hidden_size=gmm_head_cfg.hidden_size,
                num_layers=gmm_head_cfg.num_layers,
                min_std=gmm_head_cfg.min_std,
                num_modes=gmm_head_cfg.num_modes,
                low_eval_noise=gmm_head_cfg.low_eval_noise,
                activation=gmm_head_cfg.activation,
        )
        self.temporal_position_encoding_fn = SinusoidalPositionEncoding(token_size)
        self.cfg = cfg

        self.device = "cuda"
        

    def temporal_encode(self, obs):
        x = torch.cat([self.modality_encoders[modality](obs[modality]).unsqueeze(2) for modality in self.obs_shape_dict.keys()], dim=2)
        
        pos_emb = self.temporal_position_encoding_fn(x)
        # print(x.shape, pos_emb.shape)
        x = x + pos_emb.unsqueeze(1)  # (B, T, num_modality, E)
        sh = x.shape
        self.temporal_transformer.compute_mask(x.shape)

        x = rearrange(x, 'b t n e -> b (t n) e')
        x = self.temporal_transformer(x)
        x = x.reshape(*sh)
        return x[:, :, 0]  # (B, T, E)

    def forward(self, obs):
        self.train()        
        x = self.temporal_encode(obs)
        return self.gmm_head(x)
    
    def predict_action(self, data):
        data = self.preprocess_input(data)
        obs = data["obs"]
        self.eval()
        with torch.no_grad():
            x = self.temporal_encode(obs)
            dist = self.gmm_head(x[:, -1])
            action = dist.sample().detach().cpu()
        return action
    
    def use_low_noise(self):
        # Use this if you just want to pick a mode.
        self.gmm_head.low_eval_noise = True
    
    def compute_loss(self, data):
        data = self.preprocess_input(data)
        obs = data["obs"]
        action = data["actions"]
        predicted_dist = self.forward(obs)
        loss = self.gmm_head.loss_fn(predicted_dist, action)
        return loss

    
    def preprocess_input(self, data):
        data = TensorUtils.to_float(TensorUtils.to_device(data, self.device))
        return data