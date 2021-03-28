import pickle
from collections import OrderedDict

import torch
import numpy as np


def encode16bit(params: OrderedDict, fname: str):
    """Transform the dtype of param to 16-bit

    Args:
      params: model state_dict
      fname: output model weight filename
    """
    custom_dict = {}
    for (name, param) in params.items():
        param = np.float64(param.cpu().numpy())
        # Skip if the param is not a np.ndarray object
        if isinstance(param, np.ndarray):
            custom_dict[name] = np.float16(param)
        else:
            custom_dict[name] = param

    # pickle.dump(custom_dict, open(fname, "wb"))
    np.savez_compressed(fname, weights=custom_dict)
    # End


def decode16bit(fname: str):
    """Read params from fname and convert to torch.tensor

    Args:
      fname: file which store the compressed params
    """
    # params = pickle.load(open(fname, "rb"))
    params = np.load(fname, allow_pickle=True)["weights"].tolist()
    custom_dict = {}
    for (name, param) in params.items():
        param = torch.tensor(param)
        custom_dict[name] = param

    return custom_dict


def encode8bit(params: OrderedDict, fname: str):
    """Convert the dtype of param to 8-bit

    Args:
      params: model state_dict
      fname: output model weight filename
    """
    custom_dict = {}
    for (name, param) in params.items():
        param = np.float64(param.cpu().numpy())
        if isinstance(param, np.ndarray):
            min_val = np.min(param)
            max_val = np.max(param)
            # Check the value of max and min
            if max_val > min_val:
                param = np.round((param - min_val) / (max_val - min_val) * 255)
            else:
                param = np.zeros_like(param)

            param = np.uint8(param)
            custom_dict[name] = (min_val, max_val, param)
        else:
            custom_dict[name] = param

    pickle.dump(custom_dict, open(fname, "wb"))


def decode8bit(fname: str):
    """Read params from fname and convert to torch.tensor

    Args:
      fname: file which store the compressed params
    """
    params = pickle.load(open(fname, "rb"))
    custom_dict = {}
    for (name, param) in params.items():
        if isinstance(param, tuple):
            min_val, max_val, param = param
            param = np.float64(param)
            param = (param / 255 * (max_val - min_val)) + min_val
            param = torch.tensor(param)
        else:
            param = torch.tensor(param)

        custom_dict[name] = param

    return custom_dict
