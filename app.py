import torch
import sys
import os
import subprocess
import uuid
import gradio as gr
import cv2
from omegaconf import OmegaConf

# Load default config
config = OmegaConf.load("config/infer.yaml")

def infer(
    lq_sequence, 
    task_name,
    target_width=config.data.width,
    target_height=None,  # Will be calculated based on aspect ratio
    num_inference_steps=config.num_inference_steps,
    decode_chunk_size=config.decode_chunk_size,
    overlap=config.overlap,
    noise_aug_strength=config.noise_aug_strength,
    min_guidance_scale=config.min_appearance_guidance_scale,
    max_guidance_scale=config.max_appearance_guidance_scale,
    i2i_noise_strength=config.i2i_noise_strength,
    frames_per_batch=config.data.n_sample_frames,
    seed=77,
    progress=gr.Progress()
):
    status = gr.Markdown("Initializing...")
    
    try:
        # Task selection with progress
        if task_name == "BFR":
            task_id = "0"
            status.value = "🎯 Task: Blind Face Restoration selected"
        elif task_name == "colorization":
            task_id = "1"
            status.value = "🎯 Task: Colorization selected"
        elif task_name == "BFR + colorization":
            task_id = "0,1"
            status.value = "🎯 Task: Combined BFR + Colorization selected"

        # Video preprocessing with progress
        status.value += "\n📥 Loading video..."
        cap = cv2.VideoCapture(lq_sequence)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate height based on aspect ratio if not provided
        if target_height is None or target_height == 0:
            target_height = int((target_width / orig_width) * orig_height)
            # Ensure height is divisible by 8
            target_height = target_height - (target_height % 8)
        
        status.value += f"\n📊 Original video dimensions: {orig_width}x{orig_height}"
        status.value += f"\n📊 Target video dimensions: {target_width}x{target_height}"
        status.value += f"\n📊 Total frames: {total_frames}"
        
        # Configuration display
        status.value += "\n\n⚙️ Processing Configuration:"
        status.value += f"\n• Inference steps: {num_inference_steps}"
        status.value += f"\n• Decode chunk size: {decode_chunk_size}"
        status.value += f"\n• Frame overlap: {overlap}"
        status.value += f"\n• Noise augmentation strength: {noise_aug_strength}"
        status.value += f"\n• Guidance scale range: {min_guidance_scale} → {max_guidance_scale}"
        status.value += f"\n• Image-to-image noise strength: {i2i_noise_strength}"
        status.value += f"\n• Frames per batch: {frames_per_batch}"
        status.value += f"\n• Random seed: {seed}"
        
        # Create temporary config with user parameters
        output_dir = f"results_{str(uuid.uuid4())}"
        user_config = {
            "data": {
                "width": target_width,
                "height": target_height,
                "n_sample_frames": frames_per_batch
            },
            "num_inference_steps": num_inference_steps,
            "decode_chunk_size": decode_chunk_size,
            "overlap": overlap,
            "noise_aug_strength": noise_aug_strength,
            "min_appearance_guidance_scale": min_guidance_scale,
            "max_appearance_guidance_scale": max_guidance_scale,
            "i2i_noise_strength": i2i_noise_strength,
            "weight_dtype": config.weight_dtype,
            "pretrained_model_name_or_path": config.pretrained_model_name_or_path,
            "unet_checkpoint_path": config.unet_checkpoint_path,
            "id_linear_checkpoint_path": config.id_linear_checkpoint_path,
            "net_arcface_checkpoint_path": config.net_arcface_checkpoint_path
        }
        
        temp_config_path = f"config/temp_{str(uuid.uuid4())}.yaml"
        OmegaConf.save(config=user_config, f=temp_config_path)
        
        # Model inference setup
        status.value += "\n\n🚀 Starting video processing..."
        command = [
            sys.executable,
            "infer.py",
            "--config", temp_config_path,
            "--task_ids", f"{task_id}",
            "--input_path", f"{lq_sequence}",
            "--output_dir", output_dir,
            "--seed", str(seed)
        ]

        # Process with real-time output parsing
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Track different processing stages
        stages = {
            "Generating the first clip": 0.4,
            "Generating all video clips": 0.6,
            "Processing frame": 0.8,
            "Saving video": 0.9
        }

        # Process output
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.strip()
                print(line, flush=True)  # Print to terminal
                
                # Update progress based on recognized patterns
                if "Frame preprocessing" in line:
                    if "Frame" in line:
                        try:
                            current, total = line.split("Frame: ")[1].split("/")
                            progress(float(current) / float(total), desc="Preprocessing frames")
                            status.value += f"\n🎞️ Preprocessing frame {current}/{total}"
                        except:
                            pass
                
                elif "First clip inference" in line:
                    if "Step:" in line:
                        try:
                            current, total = line.split("Step: ")[1].split("/")
                            progress(float(current) / float(total), desc="First clip generation")
                            status.value += f"\n🔄 First clip - Step {current}/{total}"
                        except:
                            pass
                
                elif "Full video generation" in line:
                    if "Step:" in line:
                        try:
                            current, total = line.split("Step: ")[1].split("/")
                            progress(float(current) / float(total), desc="Full video generation")
                            status.value += f"\n🎬 Full video - Step {current}/{total}"
                        except:
                            pass
                
                # Always append the raw line for complete logging
                status.value += f"\n{line}"
                
                # Update overall progress based on stages
                for stage_text, progress_value in stages.items():
                    if stage_text in line:
                        progress(progress_value, desc=stage_text)
                        status.value += f"\n📍 Stage: {stage_text}"

        # Cleanup temporary config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

        # Check for errors
        if process.returncode != 0:
            error_output = process.stderr.read()
            raise Exception(f"Error during processing: {error_output}")

        # Find the output video
        output_files = [f for f in os.listdir(output_dir) if f.endswith('.mp4')]
        if not output_files:
            raise Exception("No output video found")
            
        output_path = os.path.join(output_dir, output_files[0])
        return output_path, status

    except Exception as e:
        error_msg = f"Error during processing: {str(e)}"
        status.value += f"\n❌ {error_msg}"
        print(error_msg)
        return None, status

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Video Enhancement Interface")
    
    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Input Video")
            task_name = gr.Radio(
                choices=["BFR", "colorization", "BFR + colorization"],
                value="BFR",
                label="Task"
            )
            
            with gr.Accordion("Advanced Settings", open=False):
                target_width = gr.Number(value=config.data.width, label="Target Width", precision=0)
                target_height = gr.Number(value=0, label="Target Height (Auto)", precision=0, interactive=False)
                num_inference_steps = gr.Slider(minimum=1, maximum=100, value=config.num_inference_steps, step=1, label="Inference Steps")
                decode_chunk_size = gr.Slider(minimum=1, maximum=32, value=config.decode_chunk_size, step=1, label="Decode Chunk Size")
                overlap = gr.Slider(minimum=0, maximum=16, value=config.overlap, step=1, label="Frame Overlap")
                noise_aug_strength = gr.Slider(minimum=0, maximum=1, value=config.noise_aug_strength, step=0.01, label="Noise Augmentation Strength")
                min_guidance = gr.Slider(minimum=1, maximum=10, value=config.min_appearance_guidance_scale, step=0.1, label="Min Guidance Scale")
                max_guidance = gr.Slider(minimum=1, maximum=10, value=config.max_appearance_guidance_scale, step=0.1, label="Max Guidance Scale")
                i2i_strength = gr.Slider(minimum=0, maximum=1, value=config.i2i_noise_strength, step=0.1, label="Image-to-Image Strength")
                frames_batch = gr.Slider(minimum=1, maximum=32, value=config.data.n_sample_frames, step=1, label="Frames per Batch")
                seed = gr.Number(value=77, label="Random Seed", precision=0)
            
            process_btn = gr.Button("Process Video")
        
        with gr.Column():
            output_video = gr.Video(label="Enhanced Video")
            status_output = gr.Markdown()
    
    # Process video event
    process_btn.click(
        fn=infer,
        inputs=[
            input_video, task_name, 
            target_width, target_height,
            num_inference_steps, decode_chunk_size,
            overlap, noise_aug_strength,
            min_guidance, max_guidance,
            i2i_strength, frames_batch,
            seed
        ],
        outputs=[output_video, status_output]
    )

demo.launch(share=True)