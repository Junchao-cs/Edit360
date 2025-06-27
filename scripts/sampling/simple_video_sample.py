import math
import os
import sys
from glob import glob
from pathlib import Path
from typing import List, Optional
import argparse
import tyro


sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../../")))
import cv2
import imageio
from mediapy import write_video
import numpy as np
import torch
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from PIL import Image
from rembg import remove
from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config
from torchvision.transforms import ToTensor


def sample_one(
    input_path: str = "assets/test_image_1.png",  # Can either be image file or folder with image files
    num_frames: Optional[int] = None,  # 21 for SV3D
    num_steps: Optional[int] = None,
    version: str = "sv3d_u",
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 23,
    decoding_t: int = 3,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder_mp4: Optional[str] = None,
    output_folder_img: Optional[str] = None,
    elevations_deg: Optional[float | List[float]] = 10.0,  # For SV3D
    azimuths_deg: Optional[List[float]] = None,  # For SV3D
    image_frame_ratio: Optional[float] = None,
    verbose: Optional[bool] = False,
):
    """
    CUDA_VISIBLE_DEVICES=0 \
    python scripts/sampling/simple_video_sample.py \
        --mode one \
        --input_path ./assets/test_image_1.png \
        --version sv3d_u \
        --output_folder_mp4 ./outputs/mp4 \
        --output_folder_img ./outputs/img \
        --seed 23
    """

    if version == "sv3d_u":
        num_frames = 21
        num_steps = default(num_steps, 50)
        output_folder_mp4 = default(output_folder_mp4, "./outputs/mp4")
        model_config = "scripts/sampling/configs/sv3d_u.yaml"
        cond_aug = 1e-5
    elif version == "sv3d_p":
        num_frames = 21
        num_steps = default(num_steps, 50)
        output_folder_mp4 = default(output_folder_mp4, "outputs/simple_video_sample/sv3d_p/")
        model_config = "scripts/sampling/configs/sv3d_p.yaml"
        cond_aug = 1e-5
        if isinstance(elevations_deg, float) or isinstance(elevations_deg, int):
            elevations_deg = [elevations_deg] * num_frames
        assert (
            len(elevations_deg) == num_frames
        ), f"Please provide 1 value, or a list of {num_frames} values for elevations_deg! Given {len(elevations_deg)}"
        polars_rad = [np.deg2rad(90 - e) for e in elevations_deg]
        if azimuths_deg is None:
            azimuths_deg = np.linspace(0, 360, num_frames + 1)[1:] % 360
        assert (
            len(azimuths_deg) == num_frames
        ), f"Please provide a list of {num_frames} values for azimuths_deg! Given {len(azimuths_deg)}"
        azimuths_rad = [np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg]
        azimuths_rad[:-1].sort()
    else:
        raise ValueError(f"Version {version} does not exist.")

    model, filter = load_model(
        model_config,
        device,
        num_frames,
        num_steps,
        verbose,
    )
    torch.manual_seed(seed)

    path = Path(input_path)
    all_img_paths = []
    if path.is_file():
        if any([input_path.endswith(x) for x in ["jpg", "jpeg", "png"]]):
            all_img_paths = [input_path]
        else:
            raise ValueError("Path is not valid image file.")
    elif path.is_dir():
        all_img_paths = sorted(
            [
                f
                for f in path.iterdir()
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )
        if len(all_img_paths) == 0:
            raise ValueError("Folder does not contain any images.")
    else:
        raise ValueError


    # for input_img_path in all_img_paths:

    #     input_image = Image.open(input_img_path)

    #     if input_image.mode != "RGB":
    #         input_image = input_image.convert("RGB")
            
    #     input_image = input_image.resize((576, 576), Image.Resampling.LANCZOS)
    #     image = ToTensor()(input_image)
    #     image = image * 2.0 - 1.0
    
    for input_img_path in all_img_paths:
        if "sv3d" in version:
            image = Image.open(input_img_path)
            if image.mode == "RGBA":
                pass
            else:
                # remove bg
                image.thumbnail([768, 768], Image.Resampling.LANCZOS)
                image = remove(image.convert("RGBA"), alpha_matting=True)

            # resize object in frame
            image_arr = np.array(image)
            in_w, in_h = image_arr.shape[:2]
            ret, mask = cv2.threshold(
                np.array(image.split()[-1]), 0, 255, cv2.THRESH_BINARY
            )
            x, y, w, h = cv2.boundingRect(mask)
            max_size = max(w, h)
            side_len = (
                int(max_size / image_frame_ratio)
                if image_frame_ratio is not None
                else in_w
            )
            padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
            center = side_len // 2
            padded_image[
                center - h // 2 : center - h // 2 + h,
                center - w // 2 : center - w // 2 + w,
            ] = image_arr[y : y + h, x : x + w]
            # resize frame to 576x576
            rgba = Image.fromarray(padded_image).resize((576, 576), Image.LANCZOS)
            # white bg
            rgba_arr = np.array(rgba) / 255.0
            rgb = rgba_arr[..., :3] * rgba_arr[..., -1:] + (1 - rgba_arr[..., -1:])
            input_image = Image.fromarray((rgb * 255).astype(np.uint8))


        image = ToTensor()(input_image)
        image = image * 2.0 - 1.0

        image = image.unsqueeze(0).to(device)
        H, W = image.shape[2:]
        assert image.shape[1] == 3
        F = 8
        C = 4
        shape = (num_frames, C, H // F, W // F)
        if (H, W) != (576, 1024) and "sv3d" not in version:
            print(
                "WARNING: The conditioning frame you provided is not 576x1024. This leads to suboptimal performance as model was only trained on 576x1024. Consider increasing `cond_aug`."
            )
        if (H, W) != (576, 576) and "sv3d" in version:
            print(
                "WARNING: The conditioning frame you provided is not 576x576. This leads to suboptimal performance as model was only trained on 576x576."
            )
        if motion_bucket_id > 255:
            print(
                "WARNING: High motion bucket! This may lead to suboptimal performance."
            )

        if fps_id < 5:
            print("WARNING: Small fps value! This may lead to suboptimal performance.")

        if fps_id > 30:
            print("WARNING: Large fps value! This may lead to suboptimal performance.")

        value_dict = {}
        value_dict["cond_frames_without_noise"] = image
        value_dict["motion_bucket_id"] = motion_bucket_id
        value_dict["fps_id"] = fps_id
        value_dict["cond_aug"] = cond_aug
        value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)
        if "sv3d_p" in version:
            value_dict["polars_rad"] = polars_rad
            value_dict["azimuths_rad"] = azimuths_rad

        with torch.no_grad():
            with torch.autocast(device):
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    [1, num_frames],
                    T=num_frames,
                    device=device,
                )
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=[
                        "cond_frames",
                        "cond_frames_without_noise",
                    ],
                )

                for k in ["crossattn", "concat"]:
                    uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                    uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                    c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                    c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

                randn = torch.randn(shape, device=device)
                
                additional_model_inputs = {}
                additional_model_inputs["image_only_indicator"] = torch.zeros(
                    2, num_frames
                ).to(device)
                additional_model_inputs["num_video_frames"] = batch["num_video_frames"]
                
                def denoiser(input, sigma, c, is_front_view, kv_f_list):
                      output, kv_f_list = model.denoiser( model.model, input, sigma, c, is_front_view, kv_f_list, **additional_model_inputs)
                      return output, kv_f_list

                samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)
                model.en_and_decode_n_samples_a_time = decoding_t
                samples_x = model.decode_first_stage(samples_z)

                if "sv3d" in version:
                    samples_x[-1:] = value_dict["cond_frames_without_noise"]
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

                # print(f"After clamping, shape of samples: {samples.shape}")
                os.makedirs(output_folder_mp4, exist_ok=True)
                base_count = len(glob(os.path.join(output_folder_mp4, "*.mp4")))

                imageio.imwrite(
                    os.path.join(output_folder_mp4, f"{base_count:02d}.jpg"), input_image
                )

                samples = embed_watermark(samples)
                samples = filter(samples)
                vid = (
                    (rearrange(samples, "t c h w -> t h w c") * 255)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
                
                os.makedirs(output_folder_img, exist_ok=True)


                for i, frame in enumerate(vid):
                    if i < 20:
                        frame_number = i + 2  
                    else:
                        frame_number = 1  
                    frame_image_path = os.path.join(output_folder_img, f"{frame_number}.png")
                    Image.fromarray(frame).save(frame_image_path)

                
                video_path = os.path.join(output_folder_mp4, f"{base_count:02d}.mp4")
                imageio.mimwrite(video_path, vid, fps=3, format='FFMPEG')
                

def sample_two(
    input_path_f: str = "assets/test_image_1.png",
    input_path_b: str = "assets/test_image_2.png",
    num_frames: Optional[int] = None,  # 21 for SV3D
    num_steps: Optional[int] = None,
    version: str = "sv3d_u",
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 23,
    decoding_t: int = 1,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder_mp4: Optional[str] = None,
    output_folder_img: Optional[str] = None,
    if_use_mf: bool = False,
    elevations_deg: Optional[float | List[float]] = 10.0,  # For SV3D
    azimuths_deg: Optional[List[float]] = None,  # For SV3D
    image_frame_ratio: Optional[float] = None,
    verbose: Optional[bool] = False,
    path_b_num: int = 10
):
    """
    CUDA_VISIBLE_DEVICES=0 \
    python scripts/sampling/simple_video_sample.py \
        --mode two \
        --input_path_f ./assets/test_image_1.png \
        --input_path_b ./assets/test_image_2.png \
        --version sv3d_u \
        --output_folder_mp4 ./outputs/mp4 \
        --output_folder_img ./outputs/img \
        --path_b_num 11 \
        --seed 23
    """


    if version == "sv3d_u":
        num_frames = 21
        num_steps = default(num_steps, 50)
        output_folder_mp4 = default(output_folder_mp4, "outputs/simple_video_sample/sv3d_u/")
        model_config = "scripts/sampling/configs/sv3d_u2.yaml"
        cond_aug = 1e-5
    elif version == "sv3d_p":
        num_frames = 21
        num_steps = default(num_steps, 50)
        output_folder_mp4 = default(output_folder_mp4, "outputs/simple_video_sample/sv3d_p/")
        model_config = "scripts/sampling/configs/sv3d_p2.yaml"
        cond_aug = 1e-5
        if isinstance(elevations_deg, float) or isinstance(elevations_deg, int):
            elevations_deg = [elevations_deg] * num_frames
        assert (
            len(elevations_deg) == num_frames
        ), f"Please provide 1 value, or a list of {num_frames} values for elevations_deg! Given {len(elevations_deg)}"
        polars_rad = [np.deg2rad(90 - e) for e in elevations_deg]
        if azimuths_deg is None:
            azimuths_deg = np.linspace(0, 360, num_frames + 1)[1:] % 360
        assert (
            len(azimuths_deg) == num_frames
        ), f"Please provide a list of {num_frames} values for azimuths_deg! Given {len(azimuths_deg)}"
        azimuths_rad = [np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg]
        azimuths_rad[:-1].sort()
    else:
        raise ValueError(f"Version {version} does not exist.")

    model, filter = load_model(
        model_config,
        device,
        num_frames,
        num_steps,
        verbose,
    )
    torch.manual_seed(seed)

    path_f = Path(input_path_f)
    all_img_paths_f = []
    if path_f.is_file():
        if any([input_path_f.endswith(x) for x in ["jpg", "jpeg", "png"]]):
            all_img_paths_f = [input_path_f]
        else:
            raise ValueError("Path is not valid image file.")
    elif path_f.is_dir():
        all_img_paths_f = sorted(
            [
                f
                for f in path_f.iterdir()
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )
        if len(all_img_paths_f) == 0:
            raise ValueError("Folder does not contain any images.")
    else:
        raise ValueError
    
    
    path_b = Path(input_path_b)
    all_img_paths_b = []
    if path_b.is_file():
        if any([input_path_b.endswith(x) for x in ["jpg", "jpeg", "png"]]):
            all_img_paths_b = [input_path_b]
        else:
            raise ValueError("Path is not valid image file.")
    elif path_b.is_dir():
        all_img_paths_b = sorted(
            [
                f
                for f in path_b.iterdir()
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )
        if len(all_img_paths_b) == 0:
            raise ValueError("Folder does not contain any images.")
    else:
        raise ValueError

    # for input_img_path_f in all_img_paths_f:
    #     input_image_f = Image.open(input_img_path_f)
    #     if input_image_f.mode != "RGB":
    #         input_image_f = input_image_f.convert("RGB")
            
    #     input_image_f = input_image_f.resize((576, 576), Image.Resampling.LANCZOS)
    #     image_f = ToTensor()(input_image_f)
    #     image_f = image_f * 2.0 - 1.0

    # for input_img_path_b in all_img_paths_b:
    #     input_image_b = Image.open(input_img_path_b)
    #     if input_image_b.mode != "RGB":
    #         input_image_b = input_image_b.convert("RGB")
        
    #     input_image_b = input_image_b.resize((576, 576), Image.Resampling.LANCZOS)
    #     image_b = ToTensor()(input_image_b)
    #     image_b = image_b * 2.0 - 1.0

    for input_img_path_f in all_img_paths_f:
        if "sv3d" in version:
            image_f = Image.open(input_img_path_f)
            if image_f.mode == "RGBA":
                pass
            else:
                # remove bg
                image_f.thumbnail([768, 768], Image.Resampling.LANCZOS)
                image_f = remove(image_f.convert("RGBA"), alpha_matting=True)

            # resize object in frame
            image_arr_f = np.array(image_f)
            in_w, in_h = image_arr_f.shape[:2]
            ret, mask = cv2.threshold(
                np.array(image_f.split()[-1]), 0, 255, cv2.THRESH_BINARY
            )
            x, y, w, h = cv2.boundingRect(mask)
            max_size = max(w, h)
            side_len = (
                int(max_size / image_frame_ratio)
                if image_frame_ratio is not None
                else in_w
            )
            padded_image_f = np.zeros((side_len, side_len, 4), dtype=np.uint8)
            center = side_len // 2
            padded_image_f[
                center - h // 2 : center - h // 2 + h,
                center - w // 2 : center - w // 2 + w,
            ] = image_arr_f[y : y + h, x : x + w]
            # resize frame to 576x576
            rgba_f = Image.fromarray(padded_image_f).resize((576, 576), Image.LANCZOS)
            # white bg
            rgba_arr_f = np.array(rgba_f) / 255.0
            rgb_f = rgba_arr_f[..., :3] * rgba_arr_f[..., -1:] + (1 - rgba_arr_f[..., -1:])
            input_image_f = Image.fromarray((rgb_f * 255).astype(np.uint8))

        else:
            with Image.open(input_img_path_f) as image_f:
                if image_f.mode == "RGBA":
                    input_image_f = image_f.convert("RGB")
                w, h = image_f.size

                if h % 64 != 0 or w % 64 != 0:
                    width, height = map(lambda x: x - x % 64, (w, h))
                    input_image_f = input_image_f.resize((width, height))
                    print(
                        f"WARNING: Your image_f is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
                    )

    for input_img_path_b in all_img_paths_b:
        if "sv3d" in version:
            image_b = Image.open(input_img_path_b)
            if image_b.mode == "RGBA":
                pass
            else:
                # remove bg
                image_b.thumbnail([768, 768], Image.Resampling.LANCZOS)
                image_b = remove(image_b.convert("RGBA"), alpha_matting=True)

            # resize object in frame
            image_arr_b = np.array(image_b)
            in_w, in_h = image_arr_b.shape[:2]
            ret, mask = cv2.threshold(
                np.array(image_b.split()[-1]), 0, 255, cv2.THRESH_BINARY
            )
            x, y, w, h = cv2.boundingRect(mask)
            max_size = max(w, h)
            side_len = (
                int(max_size / image_frame_ratio)
                if image_frame_ratio is not None
                else in_w
            )
            padded_image_b = np.zeros((side_len, side_len, 4), dtype=np.uint8)
            center = side_len // 2
            padded_image_b[
                center - h // 2 : center - h // 2 + h,
                center - w // 2 : center - w // 2 + w,
            ] = image_arr_b[y : y + h, x : x + w]
            # resize frame to 576x576
            rgba_b = Image.fromarray(padded_image_b).resize((576, 576), Image.LANCZOS)
            # white bg
            rgba_arr_b = np.array(rgba_b) / 255.0
            rgb_b = rgba_arr_b[..., :3] * rgba_arr_b[..., -1:] + (1 - rgba_arr_b[..., -1:])
            input_image_b = Image.fromarray((rgb_b * 255).astype(np.uint8))

        else:
            with Image.open(input_img_path_b) as image_b:
                if image_b.mode == "RGBA":
                    input_image_b = image_b.convert("RGB")
                w, h = image_b.size

                if h % 64 != 0 or w % 64 != 0:
                    width, height = map(lambda x: x - x % 64, (w, h))
                    input_image_b = input_image_b.resize((width, height))
                    print(
                        f"WARNING: Your image_b is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
                    )


        image_f = ToTensor()(input_image_f)
        image_f = image_f * 2.0 - 1.0
        
        image_b = ToTensor()(input_image_b)
        image_b = image_b * 2.0 - 1.0

        image_f = image_f.unsqueeze(0).to(device)
        image_b = image_b.unsqueeze(0).to(device)
        H, W = image_f.shape[2:]
        assert image_f.shape[1] == 3
        F = 8
        C = 4
        shape = (num_frames, C, H // F, W // F)
        if (H, W) != (576, 1024) and "sv3d" not in version:
            print(
                "WARNING: The conditioning frame you provided is not 576x1024. This leads to suboptimal performance as model was only trained on 576x1024. Consider increasing `cond_aug`."
            )
        if (H, W) != (576, 576) and "sv3d" in version:
            print(
                "WARNING: The conditioning frame you provided is not 576x576. This leads to suboptimal performance as model was only trained on 576x576."
            )
        if motion_bucket_id > 255:
            print(
                "WARNING: High motion bucket! This may lead to suboptimal performance."
            )

        if fps_id < 5:
            print("WARNING: Small fps value! This may lead to suboptimal performance.")

        if fps_id > 30:
            print("WARNING: Large fps value! This may lead to suboptimal performance.")

        value_dict_f = {}
        value_dict_f["cond_frames_without_noise"] = image_f
        value_dict_f["motion_bucket_id"] = motion_bucket_id
        value_dict_f["fps_id"] = fps_id
        value_dict_f["cond_aug"] = cond_aug
        value_dict_f["cond_frames"] = image_f + cond_aug * torch.randn_like(image_f)
        if "sv3d_p" in version:
            value_dict_f["polars_rad"] = polars_rad
            value_dict_f["azimuths_rad"] = azimuths_rad
            
        value_dict_b = {}
        value_dict_b["cond_frames_without_noise"] = image_b
        value_dict_b["motion_bucket_id"] = motion_bucket_id
        value_dict_b["fps_id"] = fps_id
        value_dict_b["cond_aug"] = cond_aug
        value_dict_b["cond_frames"] = image_b + cond_aug * torch.randn_like(image_b)
        if "sv3d_p" in version:
            value_dict_b["polars_rad"] = polars_rad
            value_dict_b["azimuths_rad"] = azimuths_rad

        with torch.no_grad():
            with torch.autocast(device):
                batch_f, batch_uc_f = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict_f,
                    [1, num_frames],
                    T=num_frames,
                    device=device,
                )
                c_f, uc_f = model.conditioner.get_unconditional_conditioning(
                    batch_f,
                    batch_uc=batch_uc_f,
                    force_uc_zero_embeddings=[
                        "cond_frames",
                        "cond_frames_without_noise",
                    ],
                )

                for k in ["crossattn", "concat"]:
                    uc_f[k] = repeat(uc_f[k], "b ... -> b t ...", t=num_frames)
                    uc_f[k] = rearrange(uc_f[k], "b t ... -> (b t) ...", t=num_frames)
                    c_f[k] = repeat(c_f[k], "b ... -> b t ...", t=num_frames)
                    c_f[k] = rearrange(c_f[k], "b t ... -> (b t) ...", t=num_frames)

                batch_b, batch_uc_b = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict_b,
                    [1, num_frames],
                    T=num_frames,
                    device=device,
                )
                c_b, uc_b = model.conditioner.get_unconditional_conditioning(
                    batch_b,
                    batch_uc=batch_uc_b,
                    force_uc_zero_embeddings=[
                        "cond_frames",
                        "cond_frames_without_noise",
                    ],
                )

                for k in ["crossattn", "concat"]:
                    uc_b[k] = repeat(uc_b[k], "b ... -> b t ...", t=num_frames)
                    uc_b[k] = rearrange(uc_b[k], "b t ... -> (b t) ...", t=num_frames)
                    c_b[k] = repeat(c_b[k], "b ... -> b t ...", t=num_frames)
                    c_b[k] = rearrange(c_b[k], "b t ... -> (b t) ...", t=num_frames)


                randn = torch.randn(shape, device=device)

                additional_model_inputs = {}
                additional_model_inputs["image_only_indicator"] = torch.zeros(
                    2, num_frames
                ).to(device)
                additional_model_inputs["num_video_frames"] = batch_f["num_video_frames"]
                
                def denoiser(input, sigma, c, is_front_view, kv_f_list):
                      output, kv_f_list = model.denoiser( model.model, input, sigma, c, is_front_view, kv_f_list, **additional_model_inputs)
                      return output, kv_f_list

                samples_z = model.sampler(denoiser, randn, cond_f=c_f, cond_b=c_b, path_b_num=path_b_num, uc_f=uc_f, uc_b=uc_b, if_use_mf=if_use_mf)
                model.en_and_decode_n_samples_a_time = decoding_t
                samples_x = model.decode_first_stage(samples_z)

                if "sv3d" in version:
                    samples_x[-1:] = value_dict_f["cond_frames_without_noise"]
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

                # print(f"After clamping, shape of samples: {samples.shape}")
                os.makedirs(output_folder_mp4, exist_ok=True)
                base_count = len(glob(os.path.join(output_folder_mp4, "*.mp4")))

                imageio.imwrite(
                    os.path.join(output_folder_mp4, f"{base_count:02d}.jpg"), input_image_f
                )
                
                imageio.imwrite(
                    os.path.join(output_folder_mp4, f"{base_count:02d}.jpg"), input_image_b
                )

                samples = embed_watermark(samples)
                samples = filter(samples)
                vid = (
                    (rearrange(samples, "t c h w -> t h w c") * 255)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
                
                os.makedirs(output_folder_img, exist_ok=True)

                for i, frame in enumerate(vid):
                    if i < 20:
                        frame_number = i + 2  
                    else:
                        frame_number = 1  
                    frame_image_path = os.path.join(output_folder_img, f"{frame_number}.png")
                    Image.fromarray(frame).save(frame_image_path)
                
                video_path = os.path.join(output_folder_mp4, f"{base_count:02d}.mp4")
                imageio.mimwrite(video_path, vid, fps=3, format='FFMPEG')


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames" or key == "cond_frames_without_noise":
            batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=N[0])
        elif key == "polars_rad" or key == "azimuths_rad":
            batch[key] = torch.tensor(value_dict[key]).to(device).repeat(N[0])
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
    verbose: bool = False,
):
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.verbose = verbose
    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()

    filter = DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter


def main():
    parser = argparse.ArgumentParser(description="Choose which sampling function to run.")
    parser.add_argument('--mode', type=str, choices=['one', 'two'], required=True, 
                        help="Choose 'one' to run sample_one, 'two' to run sample_two.")
    
    args, remaining_args = parser.parse_known_args()
    
    if args.mode == 'one':
        tyro.cli(sample_one, args=remaining_args)
    
    elif args.mode == 'two':
        tyro.cli(sample_two, args=remaining_args)

if __name__ == "__main__":
    main()