#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import os
import json
import yaml
from functools import partial

from monai.transforms import (
    Compose, LoadImaged, ConcatItemsd, DeleteItemsd, EnsureTyped,
    Orientationd, Spacingd, ScaleIntensityRangePercentilesd, SpatialPadd
)

from dinov2.eval.linear3d_reg import (
    LinearRegressor, LinearPostprocessor, create_linear_input
)
from dinov2.eval.utils import ViTAdapterFeatureWrapper, predict_reduce_tokens
from dinov2.eval.setup import setup_and_build_model_3d


# Task-specific hardcoded configuration
predict_config = {
    "model_list": ["/app/models/fold_0_sw_ch/best_val.pth",
                   "/app/models/fold_1_sw_ch/best_val.pth",
                   "/app/models/fold_2_sw_ch/best_val.pth",
                   "/app/models/fold_3_sw_ch/best_val.pth",
                   "/app/models/fold_4_sw_ch/best_val.pth",
                   ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="3DINO Inference for FOMO25")
    parser.add_argument("--t1", type=str, required=True)
    parser.add_argument("--t2", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    # DINO stuff
    parser.add_argument('--config_file', type=str, default='/app/dinov2/configs/train/vit3d_highres.yaml', help='Path to config file used during training')    
    parser.add_argument('--arch', type=str, default='vit_large', help='Model architecture (default from your training)')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size (default from your training)')
    parser.add_argument('--image_size', type=int, default=112, help='Image size used during training (matches global_crops_size)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default="/output/tmp")
    parser.add_argument('--cache_dir', type=str, default="/output/cache")

    return parser.parse_args()


def prepare_input_images(args):            
    data_dict = {"image1": args.t1, "image2": args.t2}
    
    
    keys = list(data_dict.keys())
    transforms = Compose([
        LoadImaged(keys=keys, ensure_channel_first=True),
        ConcatItemsd(keys=keys, name="image", dim=0),
        DeleteItemsd(keys=keys),
        EnsureTyped(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(1.0,) * 3,
            mode="bilinear"
        ),
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0.05, upper=99.95,
            b_min=-1.0, b_max=1.0, clip=True, channel_wise=True
        ),
        SpatialPadd(keys=["image"], spatial_size=(args.image_size,) * 3, value=-1.0),
    ])

    processed = transforms(data_dict)
    tensor = processed["image"].unsqueeze(0)  # [1, C, H, W, D]
    print(f"Input tensor shape: {tensor.shape}")
    return tensor, tensor.shape[1]



def load_config(config_path):
    """Load YAML configuration file and merge with defaults"""
    
    # Default configuration (from your default config)
    default_config = {
        'student': {
            'arch': 'vit_large_3d',
            'patch_size': 16,
            'drop_path_rate': 0.3,
            'layerscale': 1.0e-05,
            'drop_path_uniform': True,
            'pretrained_weights': '',
            'full_pretrained_weights': '',
            'ffn_layer': 'mlp',
            'block_chunks': 4,
            'qkv_bias': True,
            'proj_bias': True,
            'ffn_bias': True
        },
        'crops': {
            'global_crops_size': 96,
            'local_crops_size': 48
        },
        'train': {
            'batch_size_per_gpu': 128,
            'data_min_axis_size': 24,
            'OFFICIAL_EPOCH_LENGTH': 25
        },
        'optim': {
            'base_lr': 0.002
        },
        'evaluation': {
            'eval_period_iterations': 12500
        }
    }
    
    # Load and merge highres config if it exists
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            highres_config = yaml.safe_load(f)
        
        # Merge configurations (highres overrides default)
        def merge_configs(default, override):
            result = default.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_configs(result[key], value)
                else:
                    result[key] = value
            return result
        
        config = merge_configs(default_config, highres_config)
        print(f"Loaded config from: {config_path}")
    else:
        config = default_config
        print(f"Warning: Config file {config_path} not found. Using defaults.")
    
    return config



def build_model_and_feature_extractor(args, input_channels):
    model, autocast_dtype = setup_and_build_model_3d(args)
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)

    feature_model = ViTAdapterFeatureWrapper(
        vit_model=model,
        input_channels=input_channels,
        n_last_blocks=4,
        autocast_ctx=autocast_ctx
    ).eval().cuda()

    return feature_model, autocast_ctx


def build_regressor_from_ckpt(ckpt_path, feature_model, input_channels):
    ckpt = torch.load(ckpt_path, map_location="cuda")

    best_name = ckpt.get("iteration_metadata", {}).get("best_regressor_name")
        
    # Fallback: Read from JSON file
    if not best_name:
        out_path = "/".join(ckpt_path.split('/')[:-1])
        try:
            metrics_path = os.path.join(out_path, "results_eval_regression.json")
            with open(metrics_path, "r") as f:
                lines = f.readlines()
                for line in reversed(lines):
                    if line.strip().startswith("{\"best_regressor\""):
                        best_name = json.loads(line)["best_regressor"]["name"]
                        break
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve best regressor from JSON: {e}")

    assert best_name, "Best regressor name not found in checkpoint or metrics file."
    print(f"Using best regressor: {best_name}")

    n_blocks = 1 if "1_blocks" in best_name else 4
    avgpool = "avgpool_True" in best_name

    # Dummy tensor to compute out_dim
    dummy_tensor = torch.randn(1, input_channels, 112, 112, 112).cuda()
    with torch.no_grad():
        dummy_output = feature_model(dummy_tensor)
        out_dim = create_linear_input(dummy_output, use_n_blocks=n_blocks, use_avgpool=avgpool).shape[1]

    # Create regressor
    regressor = LinearRegressor(out_dim, n_blocks, avgpool, num_outputs=1).cuda()
    regressor.load_state_dict({
        k.replace(f"linear_regressors.regressors_dict.{best_name}.", ""): v
        for k, v in ckpt["model"].items()
        if k.startswith(f"linear_regressors.regressors_dict.{best_name}.")
    })

    # Load feature extractor weights
    feature_model.load_state_dict({
        k.replace("feature_model.", ""): v
        for k, v in ckpt["model"].items()
        if k.startswith("feature_model.")
    })

    return regressor, best_name


def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load configuration and update args with config values if needed
    print("Loading configuration...")
    print(args)
    args.opts = []  # Empty list for additional config options
    args.pretrained_weights = ""
    config = load_config(args.config_file)
    
    # Update image size from config if not explicitly set
    if args.image_size == 112:  # default value
        config_image_size = config.get('crops', {}).get('global_crops_size', 112)
        if config_image_size != 112:
            args.image_size = config_image_size
            print(f"Updated image size from config: {args.image_size}")
    
    print(f"Using image size: {args.image_size}")
    print(f"Using config file: {args.config_file}")

    # Load images and pre-process them
    input_tensor, input_channels = prepare_input_images(args)
    input_tensor = input_tensor.to(device)
    
    # Load the feature model
    feature_model, autocast_ctx = build_model_and_feature_extractor(args, input_channels)

    pred_ages = []           # Age predicted per model
    for ckpt_path in predict_config["model_list"]:
        print(f"[INFO] Loading checkpoint: {ckpt_path}")
        regressor, best_name = build_regressor_from_ckpt(ckpt_path, feature_model, input_channels)

        with torch.no_grad(): #, autocast_ctx():
            output_tokens = predict_reduce_tokens(
                backbone=feature_model,
                heads={best_name: regressor},
                x=input_tensor.float(),
                roi=(args.image_size,) * 3,
                overlap=0.5,
                sw_bs=1,
                reduce="median",
                create_linear_input_fn=create_linear_input
            )

        predictions = output_tokens[best_name]  # [1, num_classes]                
        pred_age_model = predictions[0, 0]
        pred_ages.append(pred_age_model)
        print(f"    ↳ Predicted age: {pred_age_model:.3f}")
        pred_ages.append(pred_age_model)

    print(pred_ages)
    ages_tensor  = torch.tensor(pred_ages)
    predicted_age = float(ages_tensor.median().item())
    print(f"[✓] Final ensembled age: {predicted_age:.3f}")

    return predicted_age


def main():
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    prob = predict(args)

    with open(args.output, "w") as f:
        f.write(f"{prob:.3f}")


if __name__ == "__main__":
    main()
