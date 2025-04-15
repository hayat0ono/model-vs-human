import os
import sys

import torch
import torch.nn.utils.prune as prune
import torchvision

sys.path.append('/home/ono/PycharmProjects/proj_metamer_cornet/CORnet')
import cornet


def get_model(base_model_name, device, weight="DEFAULT"):
    try:
        if base_model_name.split("_")[0] == "cornet":
            model = getattr(cornet, f"{base_model_name}")
            if base_model_name.split("_")[1] == "r":
                return model(pretrained=True, map_location=device, times=base_model_name.split("_")[2]).to(device)
            else:
                return model(pretrained=True, map_location=device).to(device)
        else:
            model = getattr(torchvision.models, base_model_name)(weights=weight)
            return model.to(device)
    except AttributeError as e:
        raise ValueError(f"Model {base_model_name} is not available: {e}")

def get_local_model(model_name, model_root, device):
    model = get_model(model_name.split("-")[0], device)
    local_weight = torch.load(os.path.join(model_root, model_name.replace("-", "/"), 'model.pth'), map_location=device)

    if not isinstance(local_weight, dict):
        raise ValueError("Loaded weights are not in dictionary format.")

    for layer_name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            if f"{layer_name}.weight_orig" in local_weight.keys() or f"{layer_name}.weight_mask" in local_weight.keys():
                prune.ln_structured(layer, name="weight", amount=1.0, n=1, dim=0)

    model.load_state_dict(local_weight)
    return model.to(device)