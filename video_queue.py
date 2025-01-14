import os
import sys
import time
import json
from pathlib import Path
import gradio as gr
from omegaconf import OmegaConf
import threading
import queue as py_queue
from datetime import datetime

# Import the main processing function
from infer import main as process_video

class VideoProcessingQueue:
    def __init__(self):
        self.queue = list()  # List for reorderable queue
        self.current_task = None
        self.processing = False
        self.completed = []
        self.failed = []
        self.lock = threading.Lock()
        
    def add_task(self, input_path, config, task_ids=None, seed=None, mask_path=None):
        """Add a video processing task to the queue"""
        task = {
            'id': len(self.queue) + len(self.completed) + len(self.failed),
            'input_path': input_path,
            'config': config,
            'task_ids': task_ids or [0],
            'seed': seed or 77,
            'mask_path': mask_path,
            'status': 'queued',
            'current_step': None,
            'added_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'start_time': None,
            'end_time': None,
            'error': None
        }
        with self.lock:
            self.queue.append(task)
        return task['id']

    def reorder_queue(self, order):
        """Reorder the queue based on new order of task IDs"""
        with self.lock:
            # Create a map of task ID to task
            task_map = {task['id']: task for task in self.queue}
            # Reorder queue based on new order
            self.queue = [task_map[task_id] for task_id in order if task_id in task_map]

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
                with self.lock:
                    if not self.queue:
                        time.sleep(1)
                        continue
                    task = self.queue[0]  # Get first task
                    self.current_task = task
                    task['status'] = 'processing'
                    task['start_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    task['current_step'] = 'Starting processing...'

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
                    task['current_step'] = 'Processing video...'
                    process_video(task['config'], args)
                    
                    with self.lock:
                        task['status'] = 'completed'
                        task['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        task['current_step'] = 'Complete'
                        self.completed.append(task)
                        self.queue.pop(0)
                        
                except Exception as e:
                    with self.lock:
                        task['status'] = 'failed'
                        task['error'] = str(e)
                        task['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        task['current_step'] = 'Failed'
                        self.failed.append(task)
                        self.queue.pop(0)
                        
                finally:
                    self.current_task = None
            except Exception as e:
                print(f"Queue processing error: {str(e)}")
                time.sleep(1)

    def stop_processing(self):
        """Stop processing the queue"""
        self.processing = False

    def get_queue_status(self):
        """Get current queue status"""
        with self.lock:
            return {
                'current': self.current_task,
                'queued': self.queue,
                'completed': self.completed,
                'failed': self.failed
            }

def create_ui():
    """Create Gradio interface for the queue system"""
    config = OmegaConf.load("./configs/infer.yaml")
    
    queue_processor = VideoProcessingQueue()
    queue_processor.start_processing()

    def add_to_queue(video_path, task_ids, seed, mask_path=None):
        if not os.path.exists(video_path):
            return f"Error: Video file not found: {video_path}", None
            
        task_id = queue_processor.add_task(
            input_path=video_path,
            config=config,
            task_ids=[int(x.strip()) for x in task_ids.split(",") if x.strip()],
            seed=int(seed),
            mask_path=mask_path if mask_path and os.path.exists(mask_path) else None
        )
        
        # Return updated queue status
        return "Added to queue", format_queue_status(queue_processor.get_queue_status())

    def format_queue_status(status):
        """Format queue status into readable text"""
        msg = []
        
        # Current task
        if status['current']:
            task = status['current']
            msg.append("🔄 Currently Processing:")
            msg.append(f"  File: {os.path.basename(task['input_path'])}")
            msg.append(f"  Step: {task['current_step']}")
            msg.append(f"  Started: {task['start_time']}")
            msg.append("")
        
        # Queued tasks
        if status['queued']:
            msg.append("📋 Queue:")
            for i, task in enumerate(status['queued'], 1):
                msg.append(f"  {i}. {os.path.basename(task['input_path'])} (Task IDs: {task['task_ids']}, Seed: {task['seed']})")
            msg.append("")
        
        # Completed tasks
        if status['completed']:
            msg.append("✅ Recently Completed:")
            for task in status['completed'][-3:]:  # Show last 3
                msg.append(f"  • {os.path.basename(task['input_path'])}")
            msg.append("")
        
        # Failed tasks
        if status['failed']:
            msg.append("❌ Failed:")
            for task in status['failed'][-3:]:  # Show last 3
                msg.append(f"  • {os.path.basename(task['input_path'])}: {task['error']}")
        
        return "\n".join(msg) if msg else "Queue is empty"

    def get_queue_status():
        return format_queue_status(queue_processor.get_queue_status())

    with gr.Blocks() as interface:
        gr.Markdown("# Video Processing Queue")
        
        with gr.Row():
            with gr.Column():
                video_input = gr.File(label="Input Video")
                task_ids = gr.Textbox(label="Task IDs (comma-separated)", value="0")
                seed = gr.Number(label="Seed", value=77)
                mask_input = gr.File(label="Mask (optional)")
                submit_btn = gr.Button("Add to Queue", variant="primary")
            
            with gr.Column():
                status_output = gr.Textbox(
                    label="Queue Status",
                    interactive=False,
                    lines=15,
                    value="Queue is empty"
                )
                refresh_btn = gr.Button("↻ Refresh Status", variant="secondary")

        # Event handlers
        submit_btn.click(
            fn=add_to_queue,
            inputs=[video_input, task_ids, seed, mask_input],
            outputs=[gr.Textbox(visible=False), status_output]
        )
        
        refresh_btn.click(
            fn=get_queue_status,
            inputs=None,
            outputs=status_output
        )

    return interface

if __name__ == "__main__":
    interface = create_ui()
    interface.launch(server_name="0.0.0.0", share=False)