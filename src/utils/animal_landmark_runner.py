# coding: utf-8

"""
Face detection and alignment using XPose.
"""

import os
import pickle
import torch
import numpy as np
from PIL import Image
from torchvision.ops import nms

from .timer import Timer
from .rprint import rlog as log
from .helper import clean_state_dict

from .dependencies.XPose import transforms as T
from .dependencies.XPose.models import build_model
from .dependencies.XPose.predefined_keypoints import *
from .dependencies.XPose.util import box_ops
from .dependencies.XPose.util.config import Config


class XPoseRunner:
    def __init__(self, model_config_path, model_checkpoint_path, embeddings_cache_path=None, cpu_only=False, **kwargs):
        self.device_id = kwargs.get("device_id", 0)
        self.flag_use_half_precision = kwargs.get("flag_use_half_precision", True)
        self.device = f"cuda:{self.device_id}" if not cpu_only and torch.cuda.is_available() else "cpu"
        self.model = self.load_animal_model(model_config_path, model_checkpoint_path, self.device)
        self.timer = Timer()

        # Load cached text embeddings for prompts
        try:
            with open(f'{embeddings_cache_path}_9.pkl', 'rb') as f:
                self.ins_text_embeddings_9, self.kpt_text_embeddings_9 = pickle.load(f)
            with open(f'{embeddings_cache_path}_68.pkl', 'rb') as f:
                self.ins_text_embeddings_68, self.kpt_text_embeddings_68 = pickle.load(f)
            print("Loaded cached embeddings from file.")
        except Exception as e:
            raise ValueError("Could not load clip embeddings. Check the file path.") from e

    def load_animal_model(self, model_config_path, model_checkpoint_path, device):
        args = Config.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location=device)
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        return model

    def load_image(self, input_image):
        image_pil = input_image.convert("RGB")
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image, _ = transform(image_pil, None)
        return image_pil, image

    def get_unipose_output(self, image, instance_text_prompt, keypoint_text_prompt, box_threshold, IoU_threshold):
        if len(keypoint_text_prompt) == 9:
            ins_emb, kpt_emb = self.ins_text_embeddings_9, self.kpt_text_embeddings_9
        elif len(keypoint_text_prompt) == 68:
            ins_emb, kpt_emb = self.ins_text_embeddings_68, self.kpt_text_embeddings_68
        else:
            raise ValueError("Unsupported number of keypoints.")

        target = {
            "instance_text_prompt": instance_text_prompt.split(','),
            "keypoint_text_prompt": keypoint_text_prompt,
            "object_embeddings_text": ins_emb.float(),
            "kpts_embeddings_text": torch.cat([
                kpt_emb.float(),
                torch.zeros(100 - kpt_emb.shape[0], 512, device=self.device)
            ]),
            "kpt_vis_text": torch.cat([
                torch.ones(kpt_emb.shape[0], device=self.device),
                torch.zeros(100 - kpt_emb.shape[0], device=self.device)
            ]),
        }

        self.model = self.model.to(self.device)
        image = image.to(self.device)

        with torch.no_grad():
            with torch.autocast(device_type=self.device[:4], dtype=torch.float16, enabled=self.flag_use_half_precision):
                outputs = self.model(image[None], [target])

        logits = outputs["pred_logits"].sigmoid()[0]
        boxes = outputs["pred_boxes"][0]
        keypoints = outputs["pred_keypoints"][0][:, :2 * len(keypoint_text_prompt)]

        scores = logits.max(dim=1)[0]
        mask = scores > box_threshold

        boxes_filt = boxes[mask]
        keypoints_filt = keypoints[mask]

        keep = nms(box_ops.box_cxcywh_to_xyxy(boxes_filt.cpu()), scores[mask].cpu(), iou_threshold=IoU_threshold)
        return boxes_filt[keep], keypoints_filt[keep]

    def run(self, input_image, instance_text_prompt, keypoint_text_example, box_threshold=0.3, IoU_threshold=0.5):
        keypoint_dict = (
            globals().get(keypoint_text_example) or
            globals().get(instance_text_prompt) or
            globals().get("animal")
        )

        keypoint_text_prompt = keypoint_dict["keypoints"]
        image_pil, image = self.load_image(input_image)
        boxes, keypoints = self.get_unipose_output(image, instance_text_prompt, keypoint_text_prompt, box_threshold, IoU_threshold)

        width, height = image_pil.size
        keypoints = keypoints[0].cpu().numpy()[:2 * len(keypoint_text_prompt)]
        x = keypoints[0::2] * width
        y = keypoints[1::2] * height
        return np.stack((x, y), axis=1)

    def warmup(self):
        self.timer.tic()
        dummy_img = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
        _ = self.run(dummy_img, 'face', 'face', box_threshold=0.0, IoU_threshold=0.0)
        elapsed = self.timer.toc()
        log(f'XPoseRunner warmup time: {elapsed:.3f}s')
