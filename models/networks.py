import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, cfg, input_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        
        self.architecture = nn.Sequential()
        for item in cfg["architecture"]:
            self.architecture.add_module(
                item[0], get_fc(item, input_dim, output_dim)
            )
    
    def forward(self, x):
        outputs = self.architecture(x)
        return outputs


def get_fc(fc_data: list, input_dim: int, output_dim: int) -> torch.nn:
    if "linear" in fc_data[0]:
        if fc_data[1] == "input_dim":
            in_features = input_dim
        else:
            in_features = fc_data[1]
        
        if fc_data[2] == "output_dim":
            out_features = output_dim
        else:
            out_features = fc_data[2]

        return nn.Linear(in_features, out_features)
    
    elif "relu" in fc_data[0]:
        return nn.ReLU()
