import torch
import sys
import os
import subprocess
import shutil
import tempfile
import uuid
import gradio as gr
from glob import glob
import time
import cv2

# Проверяем и устанавливаем необходимые зависимости
required_packages = [
    'numpy',
    'opencv-python',
    'pillow',
    'torch',
    'torchvision'
]

def check_and_install_dependencies():
    try:
        import pkg_resources
        installed_packages = {pkg.key for pkg in pkg_resources.working_set}
        
        missing_packages = [pkg for pkg in required_packages if pkg.lower() not in installed_packages]
        
        if missing_packages:
            print(f"Установка отсутствующих пакетов: {missing_packages}")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("Все зависимости установлены успешно")
        else:
            print("Все необходимые пакеты уже установлены")
    except Exception as e:
        raise RuntimeError(f"Ошибка при установке зависимостей: {str(e)}")

# Проверяем зависимости перед запуском
check_and_install_dependencies()

# Проверяем наличие необходимых моделей
required_paths = [
    os.path.join("models", "stable-video-diffusion-img2vid-xt"),
    # Добавьте другие необходимые пути моделей
]

for path in required_paths:
    if not os.path.exists(path):
        raise RuntimeError(f"Требуемая модель отсутствует: {path}. Пожалуйста, скачайте модели вручную.")

def preprocess_video(video_path):
    """Preserves aspect ratio while resizing if needed"""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # If width is larger than 1280, scale down maintaining aspect ratio
    if width > 1280:
        scale = 1280/width
        new_width = 1280
        new_height = int(height * scale)
        # Ensure height is divisible by 8
        new_height = new_height - (new_height % 8)
        
        temp_path = f"{video_path}_resized.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                            (new_width, new_height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            out.write(resized)
        
        cap.release()
        out.release()
        
        return temp_path
    
    cap.release()
    return video_path

def infer(lq_sequence, task_name):
    # Предобработка видео
    processed_video = preprocess_video(lq_sequence)
    
    unique_id = str(uuid.uuid4())
    output_dir = f"results_{unique_id}"

    if task_name == "BFR":
        task_id = "0"
    elif task_name == "colorization":
        task_id = "1"
    elif task_name == "BFR + colorization":
        task_id = "0,1"
    
    try:
        # Очищаем CUDA кэш и освобождаем память
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Принудительный сбор мусора
            import gc
            gc.collect()
            
            python_executable = sys.executable
            
            command = [
                python_executable,
                "infer.py",
                "--config", "config/infer.yaml",
                "--task_ids", f"{task_id}",
                "--input_path", f"{processed_video}",
                "--output_dir", f"{output_dir}",
            ]
            
            # Устанавливаем переменные окружения для оптимизации памяти
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join([
                os.path.dirname(os.path.abspath(__file__)),
                env.get("PYTHONPATH", "")
            ])
            
            # Более агрессивные настройки для экономии памяти
            env["CUDA_LAUNCH_BLOCKING"] = "1"
            env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,garbage_collection_threshold:0.6"
            env["CUDA_VISIBLE_DEVICES"] = "0"
            env["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
            
            # Добавляем переменные для ограничения использования VRAM
            env["PYTORCH_CUDA_MEMORY_FRACTION"] = "0.8"  # Использовать только 80% доступной VRAM
            env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
            
            process = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                env=env
            )
            
            print("Stdout:", process.stdout)
            print("Stderr:", process.stderr)
        else:
            raise gr.Error("CUDA недоступна. Требуется GPU для работы.")

        # Очищаем CUDA кэш после выполнения
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        output_video = glob(os.path.join(output_dir,"*.mp4"))
        print(output_video)
        
        if output_video:
            output_video_path = output_video[0]
            # Создаем символическую ссылку с уникальным именем
            unique_output_path = os.path.join(
                os.path.dirname(output_video_path),
                f"{os.path.splitext(os.path.basename(output_video_path))[0]}_{unique_id}.mp4"
            )
            try:
                os.symlink(output_video_path, unique_output_path)
            except OSError:
                # Если символическая ссылка не поддерживается, копируем файл
                shutil.copy2(output_video_path, unique_output_path)
            
            return unique_output_path
        else:
            output_video_path = None
            raise gr.Error("Не удалось создать выходное видео")

    except subprocess.CalledProcessError as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        error_msg = f"Stdout: {e.stdout}\nStderr: {e.stderr}" if hasattr(e, 'stdout') else str(e)
        raise gr.Error(f"Ошибка при обработке: {error_msg}")
    except Exception as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise gr.Error(f"Неожиданная ошибка: {str(e)}")
    finally:
        # Очищаем старые результаты
        try:
            results_dirs = glob("results_*")
            current_time = time.time()
            for dir_path in results_dirs:
                # Удаляем директории старше 1 часа
                if os.path.isdir(dir_path):
                    dir_time = os.path.getctime(dir_path)
                    if current_time - dir_time > 3600:  # 1 час в секундах
                        shutil.rmtree(dir_path, ignore_errors=True)
        except Exception as e:
            print(f"Ошибка при очистке старых результатов: {e}")

css="""
div#col-container{
    margin: 0 auto;
    max-width: 982px;
}
"""
with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# SVFR: A Unified Framework for Generalized Video Face Restoration")
        gr.Markdown("SVFR is a unified framework for face video restoration that supports tasks such as BFR, Colorization, Inpainting, and their combinations within one cohesive system.")
        gr.HTML("""
        <div style="display:flex;column-gap:4px;">
            <a href="https://github.com/wangzhiyaoo/SVFR">
                <img src='https://img.shields.io/badge/GitHub-Repo-blue'>
            </a> 
            <a href="https://wangzhiyaoo.github.io/SVFR/">
                <img src='https://img.shields.io/badge/Project-Page-green'>
            </a>
            <a href="https://arxiv.org/pdf/2501.01235">
                <img src='https://img.shields.io/badge/ArXiv-Paper-red'>
            </a>
            <a href="https://huggingface.co/spaces/fffiloni/SVFR-demo?duplicate=true">
                <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/duplicate-this-space-sm.svg" alt="Duplicate this Space">
            </a>
            <a href="https://huggingface.co/fffiloni">
                <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-me-on-HF-sm-dark.svg" alt="Follow me on HF">
            </a>
        </div>
        """)
        with gr.Row():
            with gr.Column():
                input_seq = gr.Video(label="Video LQ")
                task_name = gr.Radio(
                    label="Task", 
                    choices=["BFR", "colorization", "BFR + colorization"], 
                    value="BFR"
                )
                submit_btn = gr.Button("Submit")
            with gr.Column():
                output_res = gr.Video(label="Restored")
                gr.Examples(
                    examples = [
                        ["./assert/lq/lq1.mp4", "BFR"],
                        ["./assert/lq/lq2.mp4", "BFR + colorization"],
                        ["./assert/lq/lq3.mp4", "colorization"]
                    ],
                    inputs = [input_seq, task_name]
                )
    
    submit_btn.click(
        fn = infer,
        inputs = [input_seq, task_name],
        outputs = [output_res]
    )

demo.queue().launch(
    show_api=False,
    show_error=True,
    server_name='0.0.0.0',  # Allow access from LAN
    server_port=7860  # You can change the port if needed
)