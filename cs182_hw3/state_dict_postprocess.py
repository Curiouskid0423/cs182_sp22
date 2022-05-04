import torch as th
from collections import OrderedDict
root_folder = "./"
best_model_file = root_folder+"best_models/part1_best_model.pt"

def rename_state_dict_gradescope(path):
    """
    Stupid autograder cannot handle the case of `nn.DataParallel` call
    """
    new_model_state_dict = OrderedDict()
    ckpt = th.load(path)
    for key, val in ckpt['model_state_dict'].items():
        key = key[7:] # remove the word "module."
        new_model_state_dict[key] = val
    ckpt['model_state_dict'] = new_model_state_dict
    th.save(ckpt, best_model_file)
    print(f"Successfully saved at {best_model_file}\nNew keys of state dict:")
    print(ckpt['model_state_dict'].keys())


if __name__ == '__main__':
    rename_state_dict_gradescope(
        root_folder+"best_models/part1_best_model_multiGPU.pt"
        )