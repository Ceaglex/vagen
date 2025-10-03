import torch
from prodigyopt import Prodigy
from safetensors import safe_open


def print_params(transformer):
    trainable_params = 0
    total_params = 0
    for name, param in transformer.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
        total_params += param.numel()
    print(f"=== 总参数 {total_params} ==== 可训练参数 {trainable_params} ===")



def init_weights(module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
            # torch.nn.init.xavier_normal_(module.bias)
    elif isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Conv3d) or isinstance(module, torch.nn.Conv1d):
        torch.nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    # elif isinstance(module, torch.nn.LayerNorm):
    #     torch.nn.init.ones_(module.weight)
    #     torch.nn.init.zeros_(module.bias)


def load_weights(module, safetensor_path):
    if safetensor_path is not None:
        state_dict = {}
        with safe_open(safetensor_path, framework="pt", device="cpu") as f: 
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        module.load_state_dict(state_dict, strict=True)


def init_class(class_name, 
               load_dtype,
               path = None,
               subfolder = None,
               set_eval = False):
        if path is None:
            return None
        
        kwargs = {
            'torch_dtype': load_dtype,
            'local_files_only': True, 
            'use_safetensors': True
        }
        if subfolder is not None:
            kwargs['subfolder'] = subfolder
            
        module = class_name.from_pretrained(path, **kwargs)
        if set_eval:
            module.eval().requires_grad_(False)
        return module


def get_optimizer(args, params_to_optimize, use_deepspeed):
    if use_deepspeed:
        return None

    optimizer_type = getattr(args, "optimizer", "adamw").lower()
    lr = getattr(args, "learning_rate", 1e-4)
    weight_decay = getattr(args, "adam_weight_decay", 0.01)
    beta1 = getattr(args, "adam_beta1", 0.9)
    beta2 = getattr(args, "adam_beta2", 0.999)
    eps = getattr(args, "adam_epsilon", 1e-8)

    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            params_to_optimize, lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay
        )
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            params_to_optimize, lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay
        )
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(params_to_optimize, lr=lr, weight_decay=weight_decay)

    elif optimizer_type == "prodigy":
        optimizer = Prodigy(
            params_to_optimize,
            lr=lr,
            betas=(beta1, beta2, args.prodigy_beta3),
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
            weight_decay=weight_decay,
            eps=eps,
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    return optimizer



def set_requires_grad(model, target_params, print_param = False):
    for name, param in model.named_parameters():
        for target in target_params:
            if target in name:
                param.requires_grad = True  # 设置为需要梯度
                if print_param:
                    print(f"{target}", end = " ")
    print("\n")
    return model