from model_def import ECNet
from torch.nn.utils import prune
import os
import torch as t
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def prune_model(model, amount):
    modules = list(model.state_dict().items())
    param_to_prun = []
    for name, module in modules:
        name = name.split(".")
        if name[-1] == ("weight"):
            input_module = model
            for sub_mou in name[:-1]:
                input_module = input_module.__getattr__(sub_mou)
            # print(".".join(name) + ":", float(t.sum(input_module.weight == 0)) / input_module.weight.nelement())
            param_to_prun.append((input_module, "weight"))
    # print("---------------------------------------------------")
    prune.global_unstructured(param_to_prun, pruning_method=prune.L1Unstructured, amount=amount)
    for name, module in modules:
        name = name.split(".")
        if name[-1] == ("weight"):
            input_module = model
            for sub_mou in name[:-1]:
                input_module = input_module.__getattr__(sub_mou)
            # print(".".join(name) + ":", float(t.sum(input_module.weight == 0)) / input_module.weight.nelement())
            prune.remove(input_module, "weight")
    return model


if __name__ == "__main__":
    laplacian_level_count = 4
    layer_count_of_every_unet = [4, 3, 3, 3]
    first_layer_out_channels_of_every_unet = [24, 24, 16, 8]
    use_iaff = True
    iaff_r = 0.8
    use_psa = False
    weight_pth = r"./save_model/best.pth"
    prun_model_save_pth = r"./save_model/best_prune.pth"
    amount = 0.4
    model = ECNet(laplacian_level_count, layer_count_of_every_unet, first_layer_out_channels_of_every_unet, use_iaff,
                  iaff_r, use_psa)
    model.load_state_dict(t.load(weight_pth))
    model = model.cuda(0)
    model.eval()
    prune_model(model, amount)