import argparse
import warnings
import os
import numpy as np
import torch
import torch.utils.checkpoint
from PIL import Image
import random
import subprocess

from omegaconf import OmegaConf
from diffusers import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler
from transformers import CLIPVisionModelWithProjection
import torchvision.transforms as transforms
import torch.nn.functional as F
from src.models.svfr_adapter.unet_3d_svd_condition_ip import UNet3DConditionSVDModel

# pipeline 
from src.pipelines.pipeline import LQ2VideoLongSVDPipeline

from src.utils.util import (
    save_videos_grid,
    seed_everything,
)
from torchvision.utils import save_image

from src.models.id_proj import IDProjConvModel
from src.models import model_insightface_360k

from src.dataset.face_align.align import AlignImage

warnings.filterwarnings("ignore")

import decord
import cv2
from src.dataset.dataset import get_affine_transform, mean_face_lm5p_256

from tqdm import tqdm

BASE_DIR = '.'

def main(config,args):
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        cuda_visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
        print(f"CUDA_VISIBLE_DEVICES is set to: {cuda_visible_devices}")
    else:
        print("CUDA_VISIBLE_DEVICES is not set.")

    save_dir = f"{BASE_DIR}/{args.output_dir}"
    os.makedirs(save_dir,exist_ok=True)

    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        f"{BASE_DIR}/{config.pretrained_model_name_or_path}", 
        subfolder="vae",
        variant="fp16")
    
    val_noise_scheduler = EulerDiscreteScheduler.from_pretrained(
        f"{BASE_DIR}/{config.pretrained_model_name_or_path}", 
        subfolder="scheduler")
    
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        f"{BASE_DIR}/{config.pretrained_model_name_or_path}", 
        subfolder="image_encoder",
        variant="fp16")
    unet = UNet3DConditionSVDModel.from_pretrained(
        f"{BASE_DIR}/{config.pretrained_model_name_or_path}", 
        subfolder="unet",
        variant="fp16")
    
    weight_dir = 'models/face_align'
    det_path = os.path.join(BASE_DIR, weight_dir, 'yoloface_v5m.pt')
    align_instance = AlignImage("cuda", det_path=det_path)

    to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    import torch.nn as nn
    class InflatedConv3d(nn.Conv2d):
        def forward(self, x):
            x = super().forward(x)
            return x
    # Add ref channel
    old_weights = unet.conv_in.weight
    old_bias = unet.conv_in.bias
    new_conv1 = InflatedConv3d(
        12,
        old_weights.shape[0],
        kernel_size=unet.conv_in.kernel_size,
        stride=unet.conv_in.stride,
        padding=unet.conv_in.padding,
        bias=True if old_bias is not None else False,
    )
    param = torch.zeros((320, 4, 3, 3), requires_grad=True)
    new_conv1.weight = torch.nn.Parameter(torch.cat((old_weights, param), dim=1))
    if old_bias is not None:
        new_conv1.bias = old_bias
    unet.conv_in = new_conv1
    unet.config["in_channels"] = 12
    unet.config.in_channels = 12
    

    id_linear = IDProjConvModel(in_channels=512, out_channels=1024).to(device='cuda')

    # load pretrained weights
    unet_checkpoint_path = os.path.join(BASE_DIR, config.unet_checkpoint_path)
    unet.load_state_dict(
        torch.load(unet_checkpoint_path, map_location="cpu"),
        strict=True,
    )
    
    id_linear_checkpoint_path = os.path.join(BASE_DIR, config.id_linear_checkpoint_path)
    id_linear.load_state_dict(
        torch.load(id_linear_checkpoint_path, map_location="cpu"),
        strict=True,
    )

    net_arcface = model_insightface_360k.getarcface(f'{BASE_DIR}/{config.net_arcface_checkpoint_path}').eval().to(device="cuda")    

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif config.weight_dtype == "fp32":
        weight_dtype = torch.float32
    elif config.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    else:
        raise ValueError(
            f"Do not support weight dtype: {config.weight_dtype} during training"
        )

    image_encoder.to(weight_dtype)
    vae.to(weight_dtype)
    unet.to(weight_dtype)
    id_linear.to(weight_dtype)
    net_arcface.requires_grad_(False).to(weight_dtype) 

    pipe = LQ2VideoLongSVDPipeline(
        unet=unet,
        image_encoder=image_encoder,
        vae=vae,
        scheduler=val_noise_scheduler,
        feature_extractor=None

    )
    pipe = pipe.to("cuda", dtype=unet.dtype)

    seed_input = args.seed
    seed_everything(seed_input)

    video_path = args.input_path
    task_ids = args.task_ids
    
    if 2 in task_ids and args.mask_path is not None: 
        mask_path = args.mask_path
        mask = Image.open(mask_path).convert("L")
        mask_array = np.array(mask)

        white_positions = mask_array == 255

    print('task_ids:',task_ids)
    task_prompt = [0,0,0]
    for i in range(3):
        if i in task_ids:
            task_prompt[i] = 1
    print("task_prompt:",task_prompt)
    
    video_name = video_path.split('/')[-1]
    # print(video_name)

    if os.path.exists(os.path.join(save_dir, "result_frames", video_name[:-4])):
        print(os.path.join(save_dir, "result_frames", video_name[:-4]))
        # continue

    cap = decord.VideoReader(video_path, fault_tol=1)
    total_frames = len(cap)
    T = total_frames #
    print("total_frames:",total_frames)
    step=1
    drive_idx_start = 0
    drive_idx_list = list(range(drive_idx_start, drive_idx_start + T * step, step))
    assert len(drive_idx_list) == T

    imSameIDs = []
    vid_gt = []
    
    # Calculate height based on aspect ratio
    frame = cap[0].asnumpy()  # Get first frame
    orig_height, orig_width = frame.shape[:2]
    target_width = config.data.width
    target_height = int((target_width / orig_width) * orig_height)
    # Ensure height is divisible by 8
    target_height = target_height - (target_height % 8)
    
    print("\n📊 Video Dimensions:")
    print(f"Original dimensions: {orig_width}x{orig_height}")
    print(f"Target dimensions: {target_width}x{target_height}")
    print(f"Aspect ratio maintained: {orig_width/orig_height:.3f} → {target_width/target_height:.3f}")
    
    print("\n⚙️ Processing Configuration:")
    print(f"Weight dtype: {config.weight_dtype}")
    print(f"Inference steps: {config.num_inference_steps}")
    print(f"Decode chunk size: {config.decode_chunk_size}")
    print(f"Frame overlap: {config.overlap}")
    print(f"Noise augmentation strength: {config.noise_aug_strength}")
    print(f"Guidance scale range: {config.min_appearance_guidance_scale} → {config.max_appearance_guidance_scale}")
    print(f"Image-to-image noise strength: {config.i2i_noise_strength}")
    print(f"Frames per batch: {config.data.n_sample_frames}")
    
    # Add detailed progress for frame preprocessing
    print("\nPreprocessing frames...")
    pbar = tqdm(total=total_frames, desc="Frame preprocessing", position=0, leave=True)
    for i, drive_idx in enumerate(drive_idx_list):
        frame = cap[drive_idx].asnumpy()
        imSameID = Image.fromarray(frame)
        
        # Use calculated dimensions that maintain aspect ratio
        imSameID = imSameID.resize((target_width, target_height), Image.Resampling.LANCZOS)
        image_array = np.array(imSameID)
        if 2 in task_ids and args.mask_path is not None:
            image_array[white_positions] = [255, 255, 255]
        vid_gt.append(np.float32(image_array/255.))
        imSameIDs.append(imSameID)
        pbar.update(1)
        pbar.set_postfix({"Frame": f"{i+1}/{total_frames}"})
    pbar.close()
    print("Frame preprocessing complete")

    print("\nPreparing tensors...")
    # Ensure consistent tensor dimensions
    vid_lq = [(torch.from_numpy(frame).permute(2,0,1).contiguous() - 0.5) / 0.5 for frame in vid_gt]
    print("Tensor preparation complete")

    val_data = dict(
        pixel_values_vid_lq = torch.stack(vid_lq,dim=0),
        # pixel_values_ref_img=self.to_tensor(target_image),
        # pixel_values_ref_concat_img=self.to_tensor(imSrc2),
        task_ids=task_ids,
        task_id_input=torch.tensor(task_prompt),
        total_frames=total_frames,
    )
    
    window_overlap=0
    inter_frame_list = get_overlap_slide_window_indices(val_data["total_frames"],config.data.n_sample_frames,window_overlap)
    
    lq_frames = val_data["pixel_values_vid_lq"]
    task_ids = val_data["task_ids"]
    task_id_input = val_data["task_id_input"]
    height, width = val_data["pixel_values_vid_lq"].shape[-2:]
    
    # First clip generation
    print("\n=== Starting First Clip Generation ===")
    print(f"Processing first {len(inter_frame_list[0])} frames...")
    print(f"Using {config.num_inference_steps} inference steps")
    print(f"CUDA Memory before first clip: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
    
    print("\nDebug - Tensor shapes:")
    print(f"lq_frames shape: {lq_frames.shape}")
    print(f"first clip frames shape: {lq_frames[inter_frame_list[0]].shape}")
    print(f"first clip frames cuda shape: {lq_frames[inter_frame_list[0]].to('cuda').to(weight_dtype).shape}")
    
    print("Generating the first clip...")
    output = pipe(
        lq_frames[inter_frame_list[0]].to("cuda").to(weight_dtype),  # lq
        None,  # ref concat
        torch.zeros((1, len(inter_frame_list[0]), 49, 1024)).to("cuda").to(weight_dtype),  # encoder_hidden_states
        task_id_input.to("cuda").to(weight_dtype),
        height=height,
        width=width,
        num_frames=len(inter_frame_list[0]),
        decode_chunk_size=config.decode_chunk_size,
        noise_aug_strength=config.noise_aug_strength,
        min_guidance_scale=config.min_appearance_guidance_scale,
        max_guidance_scale=config.max_appearance_guidance_scale,
        overlap=config.overlap,
        frames_per_batch=len(inter_frame_list[0]),
        num_inference_steps=50,
        i2i_noise_strength=config.i2i_noise_strength,
    )
    
    video = output.frames
    print(f"\nDebug - video output shape: {video.shape}")
    ref_img_tensor = video[0][:,-1]
    print(f"Debug - ref_img_tensor shape: {ref_img_tensor.shape}")
    ref_img = (video[0][:,-1] * 0.5 + 0.5).clamp(0,1) * 255.
    print(f"Debug - ref_img shape before permute: {ref_img.shape}")
    ref_img = ref_img.permute(1,2,0).cpu().numpy().astype(np.uint8)

    pts5 = align_instance(ref_img[:,:,[2,1,0]], maxface=True)[0][0]

    warp_mat = get_affine_transform(pts5, mean_face_lm5p_256 * height/256)
    ref_img = cv2.warpAffine(np.array(Image.fromarray(ref_img)), warp_mat, (height, width), flags=cv2.INTER_CUBIC)
    ref_img = to_tensor(ref_img).to("cuda").to(weight_dtype)
    
    save_image(ref_img*0.5 + 0.5,f"{save_dir}/ref_img_align.png")
    
    ref_img = F.interpolate(ref_img.unsqueeze(0)[:, :, 0:224, 16:240], size=[112, 112], mode='bilinear')
    _, id_feature_conv = net_arcface(ref_img) 
    id_embedding = id_linear(id_feature_conv) 
    
    # Full video generation
    print('\n=== Starting Full Video Generation ===')
    print(f'Total frames to process: {val_data["total_frames"]}')
    print(f'Using batch size: {config.data.n_sample_frames}')
    print(f'Frame overlap: {config.overlap}')
    print(f'CUDA Memory before full generation: {torch.cuda.memory_allocated()/1024**2:.2f}MB')
    
    print('Generating all video clips...')
    print(f"\nDebug - Full video tensor shapes:")
    print(f"lq_frames shape: {lq_frames.shape}")
    print(f"lq_frames cuda shape: {lq_frames.to('cuda').to(weight_dtype).shape}")
    
    output = pipe(
        lq_frames.to("cuda").to(weight_dtype),  # Direct pass without dimension modification
        ref_img_tensor.to("cuda").to(weight_dtype),
        id_embedding.unsqueeze(1).repeat(1, len(lq_frames), 1, 1).to("cuda").to(weight_dtype),
        task_id_input.to("cuda").to(weight_dtype),
        height=height,
        width=width,
        num_frames=val_data["total_frames"],
        decode_chunk_size=config.decode_chunk_size,
        noise_aug_strength=config.noise_aug_strength,
        min_guidance_scale=config.min_appearance_guidance_scale,
        max_guidance_scale=config.max_appearance_guidance_scale,
        overlap=config.overlap,
        frames_per_batch=config.data.n_sample_frames,
        num_inference_steps=config.num_inference_steps,
        i2i_noise_strength=config.i2i_noise_strength,
    )
    video = output.frames
    video = (video * 0.5 + 0.5).clamp(0, 1)
    video = torch.cat([video.to(device="cuda")], dim=0).cpu()
    
    print("Video generation complete")
    print(f"CUDA Memory after generation: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
    
    # Get original video's FPS
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Save video with original FPS (without audio first)
    temp_output = f"{save_dir}/{video_name[:-4]}_{seed_input}_temp.mp4"
    final_output = f"{save_dir}/{video_name[:-4]}_{seed_input}.mp4"
    
    print(f"\n🎥 Saving initial video to: {temp_output}")
    print("Saving video frames...")
    save_videos_grid(video, temp_output, n_rows=1, fps=25)
    print("Initial video save complete")
    
    # Copy audio from original to processed video using ffmpeg
    try:
        print(f"\n🔍 Checking for audio in original video: {video_path}")
        
        # Check if original video has audio
        cmd = f"ffmpeg -i {video_path} -c copy -f null - 2>&1"
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        has_audio = "Stream #0:1: Audio" in output.decode()
        
        if has_audio:
            print("🔊 Audio stream found in original video")
            print(f"📝 Copying audio to processed video...")
            
            # Copy audio from original to processed video
            cmd = f'ffmpeg -i {temp_output} -i {video_path} -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 {final_output} -y'
            process = subprocess.Popen(
                cmd, 
                shell=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                print("✅ Audio copying successful")
                print(f"🗑️ Removing temporary file: {temp_output}")
                os.remove(temp_output)
            else:
                print(f"⚠️ FFmpeg warning/error output:\n{stderr}")
                raise Exception("FFmpeg processing failed")
                
        else:
            print("ℹ️ No audio stream found in original video")
            print(f"📝 Using video without audio")
            os.rename(temp_output, final_output)
            
    except Exception as e:
        error_msg = f"⚠️ Audio processing error: {str(e)}"
        print(error_msg)
        print(f"📝 Falling back to video without audio")
        os.rename(temp_output, final_output)

    print(f"✨ Final video saved to: {final_output}")

    if args.restore_frames:
        print("\n📸 Saving individual frames...")
        video = video.squeeze(0)
        frames_dir = os.path.join(save_dir, "result_frames", f"{video_name[:-4]}_{seed_input}")
        os.makedirs(frames_dir, exist_ok=True)
        
        total_frames = video.shape[1]
        with tqdm(total=total_frames, desc="Saving frames", position=0, leave=True) as pbar:
            for i in range(total_frames):
                save_frames_path = os.path.join(frames_dir, f'{i:08d}.png')
                save_image(video[:,i], save_frames_path)
                pbar.update(1)
                pbar.set_postfix({"Frame": f"{i+1}/{total_frames}"})
        
        print(f"✅ All frames saved to: {frames_dir}")


def get_overlap_slide_window_indices(video_length, window_size, window_overlap):
    inter_frame_list = []
    for j in range(0, video_length, window_size-window_overlap):
        inter_frame_list.append( [e % video_length for e in range(j, min(j + window_size, video_length))] )

    return inter_frame_list

if __name__ == "__main__":
    def parse_list(value):
        return [int(x) for x in value.split(",")]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/infer.yaml")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--task_ids", type=parse_list, default=[0])
    parser.add_argument("--input_path", type=str, default='./assert/lq/lq3.mp4')
    parser.add_argument("--mask_path", type=str, default=None)
    parser.add_argument("--restore_frames", action='store_true')

    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(config, args)
