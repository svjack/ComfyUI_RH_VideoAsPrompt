from weakref import ref
import torch
import os
from diffusers import (
    AutoencoderKLCogVideoX,
    # CogVideoXImageToVideoMOTPipeline,
    # CogVideoXTransformer3DMOTModel,
)
from diffusers.utils import export_to_video, load_video
from .pipeline_cogvideox_image2video_mot import CogVideoXImageToVideoMOTPipeline
from PIL import Image
from optimum.quanto import freeze, qint8, quantize

import folder_paths

from .cogvideox_transformer_3d_mot import CogVideoXTransformer3DMOTModel
import numpy as np
import comfy.utils

def pil_2_tensor(pil_image):
    image = np.array(pil_image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)
    return image

def tensor_2_pil(img_tensor):
    i = 255. * img_tensor.squeeze().cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    return img

class RunningHub_VideoAsPrompt_Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "type": (["CogVideoX"], ),
            }
        }

    RETURN_TYPES = ('RH_VideoAsPrompt_Pipeline', )
    FUNCTION = "load"

    CATEGORY = "RunningHub/VideoAsPrompt"

    def load(self, type):
        if type == "CogVideoX":
            return (self.load_cogvideox(), )
        return (None, )

    def load_cogvideox(self):
        model_base = os.path.join(folder_paths.models_dir, "Video-As-Prompt", "CogVideoX-5B")
        vae = AutoencoderKLCogVideoX.from_pretrained(model_base, subfolder="vae", torch_dtype=torch.bfloat16)
        transformer = CogVideoXTransformer3DMOTModel.from_pretrained(model_base, subfolder="transformer", torch_dtype=torch.bfloat16)
        pipe = CogVideoXImageToVideoMOTPipeline.from_pretrained(
            model_base, vae=vae, transformer=transformer, torch_dtype=torch.bfloat16,
        )
        if hasattr(pipe.vae, 'enable_slicing'):
            pipe.vae.enable_slicing()
        if hasattr(pipe.vae, 'enable_tiling'):
            pipe.vae.enable_tiling()

        quantize(pipe.transformer, qint8)
        freeze(pipe.transformer)

        pipe.enable_model_cpu_offload()
        return pipe

class RunningHub_VideoAsPrompt_Sampler_CogVideoX:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("RH_VideoAsPrompt_Pipeline", ),
                "image": ("IMAGE", ),
                "ref_video": ("IMAGE", ),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "prompt_mot_ref": ("STRING", {"default": "", "multiline": True}),
                "height": ("INT", {"default": 480, "min": 16, "max": 1024}),
                "width": ("INT", {"default": 720, "min": 16, "max": 1024}),
                "num_frames": ("INT", {"default": 49, "min": 1, "max": 1024}),
                # "frames_selection": ("STRING", {"default": "evenly", "choices": ["first", "evenly", "random"]}),
                # "use_dynamic_cfg": ("BOOLEAN", {"default": False}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 1000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "tooltip": "The random seed used for creating the noise."}),
            }
        }

    RETURN_TYPES = ('IMAGE', )
    FUNCTION = "sample"
    TITLE = 'RunningHub VideoAsPrompt Sampler(CogVideoX)'

    CATEGORY = "RunningHub/VideoAsPrompt"

    def sample(self, **kwargs):
        pipeline = kwargs["pipeline"]
        image = kwargs["image"]
        ref_video = kwargs["ref_video"]
        prompt = kwargs["prompt"]
        prompt_mot_ref = kwargs["prompt_mot_ref"]
        height = kwargs["height"]
        width = kwargs["width"]
        num_frames = kwargs["num_frames"]
        num_inference_steps = kwargs["num_inference_steps"]
        self.pbar = comfy.utils.ProgressBar(num_inference_steps + 2)
        # seed = kwargs["seed"]

        ref_video = [tensor_2_pil(ref_frame) for ref_frame in ref_video]
        image = tensor_2_pil(image)
        idx = torch.linspace(0, len(ref_video) - 1, num_frames).long().tolist()
        ref_frames = [ref_video[i] for i in idx]

        output_frames = pipeline(
            image=image,
            ref_videos=[ref_frames],
            prompt=prompt,
            prompt_mot_ref=[prompt_mot_ref],
            height=height,
            width=width,
            num_frames=num_frames,
            frames_selection="evenly",
            use_dynamic_cfg=True,
            num_inference_steps = num_inference_steps,
            update_func=self.update,
        ).frames[0]
        export_to_video(output_frames, "output.mp4")
        output_frames = [pil_2_tensor(output_frame) for output_frame in output_frames]
        return (output_frames, )

    def update(self):
        self.pbar.update(1)

NODE_CLASS_MAPPINGS = {
    "RunningHub VideoAsPrompt Sampler(CogVideoX)": RunningHub_VideoAsPrompt_Sampler_CogVideoX,
    "RunningHub VideoAsPrompt Loader": RunningHub_VideoAsPrompt_Loader,
}