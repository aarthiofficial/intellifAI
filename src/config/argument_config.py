# coding: utf-8

"""
All configs for user
"""

from dataclasses import dataclass
import tyro
from typing_extensions import Annotated
from typing import Optional, Literal
from .base_config import PrintableConfig, make_abs_path


@dataclass(repr=False)  # use repr from PrintableConfig
class ArgumentConfig(PrintableConfig):
    ########## input arguments ##########
    source: Annotated[str, tyro.conf.arg(aliases=["-s"])] = make_abs_path('../../assets/examples/source/s0.jpg')
    driving: Annotated[str, tyro.conf.arg(aliases=["-d"])] = make_abs_path('../../assets/examples/driving/d0.mp4')
    output_dir: Annotated[str, tyro.conf.arg(aliases=["-o"])] = 'animations/'

    ########## inference arguments ##########
    flag_use_half_precision: bool = True
    flag_crop_driving_video: bool = False
    device_id: int = 0
    flag_force_cpu: bool = False
    flag_normalize_lip: bool = False
    flag_source_video_eye_retargeting: bool = False
    flag_eye_retargeting: bool = False
    flag_lip_retargeting: bool = False
    flag_stitching: bool = True
    flag_relative_motion: bool = True
    flag_pasteback: bool = True
    flag_do_crop: bool = True
    driving_option: Literal["expression-friendly", "pose-friendly"] = "expression-friendly"
    driving_multiplier: float = 1.0
    driving_smooth_observation_variance: float = 3e-7
    audio_priority: Literal['source', 'driving'] = 'driving'
    animation_region: Literal["exp", "pose", "lip", "eyes", "all"] = "all"

    ########## source crop arguments ##########
    det_thresh: float = 0.15
    scale: float = 2.3
    vx_ratio: float = 0.0
    vy_ratio: float = -0.125
    flag_do_rot: bool = True
    source_max_dim: int = 1280
    source_division: int = 2

    ########## driving crop arguments ##########
    scale_crop_driving_video: float = 2.2
    vx_ratio_crop_driving_video: float = 0.0
    vy_ratio_crop_driving_video: float = -0.1

    ########## gradio arguments ##########
    server_port: Annotated[int, tyro.conf.arg(aliases=["-p"])] = 8890
    share: bool = False
    server_name: Optional[str] = "127.0.0.1"
    flag_do_torch_compile: bool = False
    gradio_temp_dir: Optional[str] = None
