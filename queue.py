import os
import sys
import time
import json
from pathlib import Path
import gradio as gr
from omegaconf import OmegaConf
import threading
from queue import Queue
from datetime import datetime

# Import the main processing function
from infer import main as process_video

class VideoProcessingQueue:
    def __init__(self):
        self.queue = Queue()
        self.current_task = None
        self.processing = False
        self.results = []
        self.lock = threading.Lock()
        
    def add_task(self, input_path, config, task_ids=None, seed=None, mask_path=None):
        """Add a video processing task to the queue"""
        task = {
            'input_path': input_path,
            'config': config,
            'task_ids': task_ids or [0],
            'seed': seed or 77,
            'mask_path': mask_path,
            'status': 'queued',
            'added_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'start_time': None,
            'end_time': None,
            'error': None
        }
        self.queue.put(task)
        self.results.append(task)
        return len(self.results) - 1  # Return task ID

    def start_processing(self):
        """Start processing the queue in a separate thread"""
        if not self.processing:
            self.processing = True
            thread = threading.Thread(target=self._process_queue)
            thread.daemon = True
            thread.start()

    def _process_queue(self):
        """Process videos in the queue"""
        while self.processing:
            try:
                if not self.queue.empty():
                    with self.lock:
                        task = self.queue.get()
                        self.current_task = task
                        task['status'] = 'processing'
                        task['start_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    try:
                        # Create args object with task parameters
                        class Args:
                            pass
                        args = Args()
                        args.input_path = task['input_path']
                        args.output_dir = "output"
                        args.seed = task['seed']
                        args.task_ids = task['task_ids']
                        args.mask_path = task['mask_path']
                        args.restore_frames = False

                        # Process the video
                        process_video(task['config'], args)
                        
                        with self.lock:
                            task['status'] = 'completed'
                            task['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            
                    except Exception as e:
                        with self.lock:
                            task['status'] = 'failed'
                            task['error'] = str(e)
                            task['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            
                    finally:
                        self.queue.task_done()
                        self.current_task = None
                else:
                    time.sleep(1)  # Prevent busy waiting
            except Exception as e:
                print(f"Queue processing error: {str(e)}")
                time.sleep(1)

    def stop_processing(self):
        """Stop processing the queue"""
        self.processing = False

    def get_status(self):
        """Get current queue status"""
        with self.lock:
            return {
                'queue_size': self.queue.qsize(),
                'current_task': self.current_task,
                'results': self.results
            }

def create_ui():
    """Create Gradio interface for the queue system"""
    # Load default config
    config = OmegaConf.load("./configs/infer.yaml")
    
    queue_processor = VideoProcessingQueue()
    queue_processor.start_processing()

    def add_to_queue(video_path, task_ids, seed, mask_path=None):
        if not os.path.exists(video_path):
            return f"Error: Video file not found: {video_path}"
            
        task_id = queue_processor.add_task(
            input_path=video_path,
            config=config,
            task_ids=[int(x.strip()) for x in task_ids.split(",") if x.strip()],
            seed=int(seed),
            mask_path=mask_path if mask_path and os.path.exists(mask_path) else None
        )
        return f"Added to queue. Task ID: {task_id}"

    def get_queue_status():
        status = queue_processor.get_status()
        
        # Format status message
        msg = f"Queue size: {status['queue_size']}\n\n"
        
        if status['current_task']:
            task = status['current_task']
            msg += f"Currently processing:\n"
            msg += f"File: {os.path.basename(task['input_path'])}\n"
            msg += f"Status: {task['status']}\n"
            msg += f"Started: {task['start_time']}\n\n"
        
        msg += "Recent tasks:\n"
        for task in reversed(status['results'][-5:]):  # Show last 5 tasks
            msg += f"- {os.path.basename(task['input_path'])}: {task['status']}"
            if task['error']:
                msg += f" (Error: {task['error']})"
            msg += "\n"
            
        return msg

    with gr.Blocks() as interface:
        gr.Markdown("# Video Processing Queue")
        
        with gr.Row():
            with gr.Column():
                video_input = gr.File(label="Input Video")
                task_ids = gr.Textbox(label="Task IDs (comma-separated)", value="0")
                seed = gr.Number(label="Seed", value=77)
                mask_input = gr.File(label="Mask (optional)")
                submit_btn = gr.Button("Add to Queue")
            
            with gr.Column():
                status_output = gr.Textbox(label="Queue Status", interactive=False)
                refresh_btn = gr.Button("Refresh Status")

        submit_btn.click(
            fn=add_to_queue,
            inputs=[video_input, task_ids, seed, mask_input],
            outputs=status_output
        )
        
        refresh_btn.click(
            fn=get_queue_status,
            inputs=[],
            outputs=status_output
        )

        # Auto-refresh status every 5 seconds
        status_output.update(get_queue_status, every=5)

    return interface

if __name__ == "__main__":
    interface = create_ui()
    interface.launch(server_name="0.0.0.0", share=False)
