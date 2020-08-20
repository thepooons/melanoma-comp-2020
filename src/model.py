import torch
import torch.nn as nn


def get_model(model_name='resnest50'):
    """Create a skeleton neural network

    Args:
        model_name (str, optional): The model structure to be downloaded
        from torch hub. Defaults to 'resnest50'.

    Returns:
        torch.nn.Module: the skeleton neural network with dummy/no weight
        tensors
    """
    net = torch.hub.load(
        'zhanghang1989/ResNeSt',
        model_name,
        pretrained=False
        )
    in_feats = net.fc.in_features
    net.fc = nn.Linear(in_features=in_feats, out_features=1)

    return net


def load_weights(model, model_ckpt_path):
    """Load the weights from a pytorch_lightning.nnModule

    Args:
        model (torch.nn.Module): the skeleton torch mode
        model_ckpt_path (str): path to the trained model's weights in `.ckpt`
        file

    Returns:
        torch.nn.Module: the model with weights loaded from the checkpoint file
    """
    model.load_state_dict(
        torch.load(
            model_ckpt_path,
            map_location=lambda storage, loc: storage
            )['state_dict'],
        strict=False
        )
    model.eval()
    return model
