# coding: utf-8

"""
Benchmark the inference speed of each module in LivePortrait.
"""

import torch
torch._dynamo.config.suppress_errors = True  # Suppress errors and fallback to eager if needed

import yaml
import time
import numpy as np

from src.utils.helper import load_model, concat_feat
from src.config.inference_config import InferenceConfig


def initialize_inputs(batch_size=1, device_id=0):
    """
    Generate random input tensors and move them to GPU.
    """
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    feature_3d = torch.randn(batch_size, 32, 16, 64, 64, device=device, dtype=dtype)
    kp_source = torch.randn(batch_size, 21, 3, device=device, dtype=dtype)
    kp_driving = torch.randn(batch_size, 21, 3, device=device, dtype=dtype)
    source_image = torch.randn(batch_size, 3, 256, 256, device=device, dtype=dtype)
    generator_input = torch.randn(batch_size, 256, 64, 64, device=device, dtype=dtype)
    eye_close_ratio = torch.randn(batch_size, 3, device=device, dtype=dtype)
    lip_close_ratio = torch.randn(batch_size, 2, device=device, dtype=dtype)

    feat_stitching = concat_feat(kp_source, kp_driving).to(dtype)
    feat_eye = concat_feat(kp_source, eye_close_ratio).to(dtype)
    feat_lip = concat_feat(kp_source, lip_close_ratio).to(dtype)

    return {
        'feature_3d': feature_3d,
        'kp_source': kp_source,
        'kp_driving': kp_driving,
        'source_image': source_image,
        'generator_input': generator_input,
        'feat_stitching': feat_stitching,
        'feat_eye': feat_eye,
        'feat_lip': feat_lip
    }


def load_and_compile_models(cfg, model_config):
    """
    Load and compile models for inference.
    """
    # Load all required modules
    appearance_feature_extractor = load_model(cfg.checkpoint_F, model_config, cfg.device_id, 'appearance_feature_extractor')
    motion_extractor = load_model(cfg.checkpoint_M, model_config, cfg.device_id, 'motion_extractor')
    warping_module = load_model(cfg.checkpoint_W, model_config, cfg.device_id, 'warping_module')
    spade_generator = load_model(cfg.checkpoint_G, model_config, cfg.device_id, 'spade_generator')
    stitching_retargeting_module = load_model(cfg.checkpoint_S, model_config, cfg.device_id, 'stitching_retargeting_module')

    # Compile inference models
    compiled_models = {
        'Appearance Feature Extractor': torch.compile(appearance_feature_extractor.half().eval(), mode='max-autotune'),
        'Motion Extractor': torch.compile(motion_extractor.half().eval(), mode='max-autotune'),
        'Warping Network': torch.compile(warping_module.half().eval(), mode='max-autotune'),
        'SPADE Decoder': torch.compile(spade_generator.half().eval(), mode='max-autotune')
    }

    for k in ['stitching', 'eye', 'lip']:
        stitching_retargeting_module[k] = torch.compile(stitching_retargeting_module[k].half().eval(), mode='max-autotune')

    return compiled_models, stitching_retargeting_module


def warm_up_models(compiled_models, stitching_retargeting_module, inputs):
    """
    Run a few inference steps to warm up models.
    """
    print("Warm up start...")
    with torch.no_grad():
        for _ in range(10):
            compiled_models['Appearance Feature Extractor'](inputs['source_image'])
            compiled_models['Motion Extractor'](inputs['source_image'])
            compiled_models['Warping Network'](inputs['feature_3d'], inputs['kp_driving'], inputs['kp_source'])
            compiled_models['SPADE Decoder'](inputs['generator_input'])
            stitching_retargeting_module['stitching'](inputs['feat_stitching'])
            stitching_retargeting_module['eye'](inputs['feat_eye'])
            stitching_retargeting_module['lip'](inputs['feat_lip'])
    print("Warm up done!")


def measure_inference_times(compiled_models, stitching_retargeting_module, inputs):
    """
    Benchmark inference time of each module.
    """
    times = {name: [] for name in compiled_models}
    times['Stitching and Retargeting Modules'] = []
    overall_times = []

    with torch.no_grad():
        for _ in range(100):
            torch.cuda.synchronize()
            overall_start = time.time()

            for name, model in compiled_models.items():
                torch.cuda.synchronize()
                start = time.time()
                if name == 'Warping Network':
                    model(inputs['feature_3d'], inputs['kp_driving'], inputs['kp_source'])
                elif name == 'SPADE Decoder':
                    model(inputs['generator_input'])
                else:
                    model(inputs['source_image'])
                torch.cuda.synchronize()
                times[name].append(time.time() - start)

            torch.cuda.synchronize()
            start = time.time()
            stitching_retargeting_module['stitching'](inputs['feat_stitching'])
            stitching_retargeting_module['eye'](inputs['feat_eye'])
            stitching_retargeting_module['lip'](inputs['feat_lip'])
            torch.cuda.synchronize()
            times['Stitching and Retargeting Modules'].append(time.time() - start)

            overall_times.append(time.time() - overall_start)

    return times, overall_times


def print_benchmark_results(compiled_models, stitching_retargeting_module, retargeting_models, times, overall_times):
    """
    Print average inference time and model parameter sizes.
    """
    print("\n--- Model Size & Timing Results ---\n")
    for name, model in compiled_models.items():
        num_params = sum(p.numel() for p in model.parameters())
        print(f"[{name}] Parameters: {num_params / 1e6:.2f} M")

    for i, key in enumerate(retargeting_models):
        num_params = sum(p.numel() for p in stitching_retargeting_module[key].parameters())
        print(f"[Stitching-{key}] Parameters: {num_params / 1e6:.2f} M")

    for name, tlist in times.items():
        print(f"{name} - Avg: {np.mean(tlist) * 1000:.2f} ms, Std: {np.std(tlist) * 1000:.2f} ms")

    print(f"\nOverall Avg Inference Time: {np.mean(overall_times) * 1000:.2f} ms")


def main():
    """
    Entry point: load configs, models, benchmark, and print results.
    """
    cfg = InferenceConfig()
    with open(cfg.models_config, 'r') as file:
        model_config = yaml.safe_load(file)

    inputs = initialize_inputs(device_id=cfg.device_id)
    compiled_models, stitching_retargeting_module = load_and_compile_models(cfg, model_config)

    warm_up_models(compiled_models, stitching_retargeting_module, inputs)
    times, overall_times = measure_inference_times(compiled_models, stitching_retargeting_module, inputs)

    print_benchmark_results(compiled_models, stitching_retargeting_module, ['stitching', 'eye', 'lip'], times, overall_times)


if __name__ == "__main__":
    main()
