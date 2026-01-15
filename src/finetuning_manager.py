"""
FineTuning Manager - Docker-based Unsloth Fine-Tuning
Uses the official unsloth/unsloth Docker container for training.
"""

import os
import json
import subprocess
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning a model."""
    # Model settings
    model_name: str = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
    max_seq_length: int = 2048
    load_in_4bit: bool = True  # QLoRA (4-bit) vs LoRA (16-bit)
    use_qlora: bool = True  # Explicit QLoRA toggle
    
    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training settings
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 10
    max_steps: int = -1
    
    # Output settings
    output_dir: str = "models"
    custom_model_name: str = ""  # User-defined model name
    logging_steps: int = 25
    
    # Dataset settings
    dataset_format: str = "alpaca"  # alpaca or chatml


# Popular Unsloth models for fine-tuning (Updated 2025)
UNSLOTH_MODELS = [
    # ===== RECOMMENDED FOR RTX 3070 Ti (8GB) =====
    # Llama 3.2 (Small - Best for testing)
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    
    # Qwen 3 (Latest - Great performance)
    "unsloth/Qwen3-0.6B-bnb-4bit",
    "unsloth/Qwen3-1.7B-bnb-4bit",
    "unsloth/Qwen3-4B-bnb-4bit",
    
    # Qwen 2.5 (Stable)
    "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    
    # Gemma 2 (Google)
    "unsloth/gemma-2-2b-it-bnb-4bit",
    
    # Phi-4 (Microsoft - Reasoning capable)
    "unsloth/Phi-4-bnb-4bit",
    
    # SmolLM 2 (Tiny but capable)
    "unsloth/SmolLM2-360M-Instruct-bnb-4bit",
    "unsloth/SmolLM2-1.7B-Instruct-bnb-4bit",
    
    # DeepSeek R1 Distill (Reasoning)
    "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit",
    
    # ===== LARGER MODELS (12GB+ VRAM) =====
    # Llama 3.1 (8B - Production)
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    
    # Llama 3.3 (70B - High-end)
    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
    
    # Qwen 3 (Larger)
    "unsloth/Qwen3-8B-bnb-4bit",
    "unsloth/Qwen3-14B-bnb-4bit",
    
    # Gemma 2 (Larger)
    "unsloth/gemma-2-9b-it-bnb-4bit",
    
    # Mistral (High quality)
    "unsloth/Mistral-Small-24B-Instruct-2501-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    
    # DeepSeek R1 (Larger)
    "unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit",
    "unsloth/DeepSeek-R1-Distill-Llama-8B-bnb-4bit",
]


class DockerFineTuningManager:
    """Manages fine-tuning using Docker container."""
    
    DOCKER_IMAGE = "unsloth/unsloth"
    CONTAINER_NAME = "modelforge-unsloth"
    JUPYTER_PORT = 8888
    JUPYTER_PASSWORD = "modelforge"
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.container_id = None
        self.workspace_path = Path.cwd().absolute()
    
    @staticmethod
    def check_docker() -> Tuple[bool, str]:
        """Check if Docker is available and running."""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                return False, "Docker is not installed"
            
            # Check if Docker daemon is running
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                return False, "Docker daemon is not running. Please start Docker Desktop."
            
            return True, f"Docker is ready"
        except FileNotFoundError:
            return False, "Docker is not installed. Please install Docker Desktop."
        except subprocess.TimeoutExpired:
            return False, "Docker is not responding. Please restart Docker Desktop."
        except Exception as e:
            return False, f"Docker error: {str(e)}"
    
    @staticmethod
    def check_nvidia_docker() -> Tuple[bool, str]:
        """Check if NVIDIA Container Toolkit is available."""
        try:
            result = subprocess.run(
                ["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:12.1.0-base-ubuntu22.04", "nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=120
            )
            if result.returncode == 0:
                gpu_name = result.stdout.strip()
                return True, f"GPU available: {gpu_name}"
            else:
                return False, "NVIDIA Container Toolkit not working. Enable GPU support in Docker Desktop settings."
        except subprocess.TimeoutExpired:
            return False, "GPU check timed out"
        except Exception as e:
            return False, f"Error checking NVIDIA Docker: {str(e)}"
    
    @staticmethod
    def check_unsloth_image() -> Tuple[bool, str]:
        """Check if unsloth Docker image is available."""
        try:
            result = subprocess.run(
                ["docker", "images", "-q", "unsloth/unsloth"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.stdout.strip():
                return True, "Unsloth image is available"
            else:
                return False, "Unsloth image not found. Click 'Pull Image' to download."
        except Exception as e:
            return False, f"Error checking image: {str(e)}"
    
    @staticmethod
    def pull_image() -> Tuple[bool, str]:
        """Pull the Unsloth Docker image."""
        try:
            result = subprocess.run(
                ["docker", "pull", "unsloth/unsloth"],
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            if result.returncode == 0:
                return True, "Unsloth image pulled successfully"
            else:
                return False, f"Failed to pull image: {result.stderr}"
        except subprocess.TimeoutExpired:
            return False, "Image pull timed out"
        except Exception as e:
            return False, f"Error pulling image: {str(e)}"
    
    def get_running_container(self) -> Optional[str]:
        """Get the ID of a running ModelForge Unsloth container."""
        try:
            result = subprocess.run(
                ["docker", "ps", "-q", "-f", f"name={self.CONTAINER_NAME}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            container_id = result.stdout.strip()
            return container_id if container_id else None
        except Exception:
            return None
    
    def stop_container(self) -> bool:
        """Stop the running container."""
        try:
            container_id = self.get_running_container()
            if container_id:
                subprocess.run(
                    ["docker", "stop", container_id],
                    capture_output=True,
                    timeout=30
                )
                subprocess.run(
                    ["docker", "rm", container_id],
                    capture_output=True,
                    timeout=30
                )
            return True
        except Exception:
            return False
    
    def start_container(self, jupyter_port: int = 8888) -> Tuple[bool, str]:
        """Start the Unsloth container with GPU support."""
        try:
            # Stop any existing container
            self.stop_container()
            
            # Create work directory if not exists
            work_dir = self.workspace_path / "work"
            work_dir.mkdir(exist_ok=True)
            
            # Copy data to work directory
            data_dir = self.workspace_path / "data"
            if data_dir.exists():
                import shutil
                work_data_dir = work_dir / "data"
                if work_data_dir.exists():
                    shutil.rmtree(work_data_dir)
                shutil.copytree(data_dir, work_data_dir)
            
            # Create models output directory
            models_dir = work_dir / "models"
            models_dir.mkdir(exist_ok=True)
            
            # Build docker run command
            cmd = [
                "docker", "run", "-d",
                "--name", self.CONTAINER_NAME,
                "-e", f"JUPYTER_PASSWORD={self.JUPYTER_PASSWORD}",
                "-p", f"{jupyter_port}:8888",
                "-v", f"{work_dir}:/workspace/work",
                "--gpus", "all",
                self.DOCKER_IMAGE
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                return False, f"Failed to start container: {result.stderr}"
            
            self.container_id = result.stdout.strip()[:12]
            return True, f"Container started! Access Jupyter Lab at http://localhost:{jupyter_port} (password: {self.JUPYTER_PASSWORD})"
        except Exception as e:
            return False, f"Error starting container: {str(e)}"
    
    def generate_training_notebook(self, dataset_path: str) -> str:
        """Generate a training notebook for the container."""
        dataset_name = Path(dataset_path).name
        
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# 🔥 ModelForge Fine-Tuning\n",
                        f"**Model:** `{self.config.model_name}`\n\n",
                        f"**Dataset:** `{dataset_name}`\n\n",
                        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n",
                        "Run each cell in order to fine-tune your model."
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Cell 1: Import Unsloth\n",
                        "from unsloth import FastLanguageModel\n",
                        "import torch\n",
                        "print(f'PyTorch: {torch.__version__}')\n",
                        "print(f'CUDA: {torch.cuda.is_available()}')\n",
                        "if torch.cuda.is_available():\n",
                        "    print(f'GPU: {torch.cuda.get_device_name(0)}')"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Cell 2: Configuration\n",
                        f'MODEL_NAME = "{self.config.model_name}"\n',
                        f"MAX_SEQ_LENGTH = {self.config.max_seq_length}\n",
                        f"LOAD_IN_4BIT = {self.config.load_in_4bit}\n",
                        f'DATASET_PATH = "/workspace/work/data/{dataset_name}"\n',
                        f'OUTPUT_DIR = "/workspace/work/models/{datetime.now().strftime("%Y%m%d_%H%M")}"\n',
                        "\n",
                        "# LoRA Config\n",
                        f"LORA_R = {self.config.lora_r}\n",
                        f"LORA_ALPHA = {self.config.lora_alpha}\n",
                        f"LORA_DROPOUT = {self.config.lora_dropout}\n",
                        "\n",
                        "# Training Config\n",
                        f"NUM_EPOCHS = {self.config.num_train_epochs}\n",
                        f"BATCH_SIZE = {self.config.per_device_train_batch_size}\n",
                        f"GRAD_ACCUM = {self.config.gradient_accumulation_steps}\n",
                        f"LEARNING_RATE = {self.config.learning_rate}\n",
                        f"WARMUP_STEPS = {self.config.warmup_steps}\n",
                        "\n",
                        "print('Configuration loaded!')"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Cell 3: Load Model\n",
                        "print(f'Loading {MODEL_NAME}...')\n",
                        "model, tokenizer = FastLanguageModel.from_pretrained(\n",
                        "    model_name=MODEL_NAME,\n",
                        "    max_seq_length=MAX_SEQ_LENGTH,\n",
                        "    load_in_4bit=LOAD_IN_4BIT,\n",
                        ")\n",
                        "print('Model loaded!')"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Cell 4: Apply LoRA\n",
                        "print('Applying LoRA adapters...')\n",
                        "model = FastLanguageModel.get_peft_model(\n",
                        "    model,\n",
                        "    r=LORA_R,\n",
                        "    lora_alpha=LORA_ALPHA,\n",
                        "    lora_dropout=LORA_DROPOUT,\n",
                        "    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',\n",
                        "                    'gate_proj', 'up_proj', 'down_proj'],\n",
                        "    bias='none',\n",
                        "    use_gradient_checkpointing='unsloth',\n",
                        "    random_state=3407,\n",
                        ")\n",
                        "print('LoRA applied!')"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Cell 5: Load Dataset\n",
                        "from datasets import load_dataset\n",
                        "\n",
                        "print(f'Loading dataset from {DATASET_PATH}...')\n",
                        "dataset = load_dataset('json', data_files=DATASET_PATH, split='train')\n",
                        "print(f'Dataset loaded: {len(dataset)} samples')\n",
                        "print('\\nFirst sample:')\n",
                        "print(dataset[0])"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Cell 6: Format Dataset (Alpaca style)\n",
                        "alpaca_prompt = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n",
                        "\n",
                        "### Instruction:\n",
                        "{instruction}\n",
                        "\n",
                        "### Input:\n",
                        "{input}\n",
                        "\n",
                        "### Response:\n",
                        "{output}'''\n",
                        "\n",
                        "def format_prompts(examples):\n",
                        "    instructions = examples['instruction']\n",
                        "    inputs = examples.get('input', [''] * len(instructions))\n",
                        "    if inputs is None:\n",
                        "        inputs = [''] * len(instructions)\n",
                        "    outputs = examples['output']\n",
                        "    texts = []\n",
                        "    for inst, inp, out in zip(instructions, inputs, outputs):\n",
                        "        text = alpaca_prompt.format(\n",
                        "            instruction=inst,\n",
                        "            input=inp if inp else '',\n",
                        "            output=out\n",
                        "        ) + tokenizer.eos_token\n",
                        "        texts.append(text)\n",
                        "    return {'text': texts}\n",
                        "\n",
                        "dataset = dataset.map(format_prompts, batched=True)\n",
                        "print('Dataset formatted!')"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Cell 7: Create Trainer\n",
                        "from trl import SFTTrainer, SFTConfig\n",
                        "\n",
                        "print('Creating trainer...')\n",
                        "trainer = SFTTrainer(\n",
                        "    model=model,\n",
                        "    tokenizer=tokenizer,\n",
                        "    train_dataset=dataset,\n",
                        "    args=SFTConfig(\n",
                        "        output_dir=OUTPUT_DIR,\n",
                        "        per_device_train_batch_size=BATCH_SIZE,\n",
                        "        gradient_accumulation_steps=GRAD_ACCUM,\n",
                        "        warmup_steps=WARMUP_STEPS,\n",
                        "        num_train_epochs=NUM_EPOCHS,\n",
                        "        learning_rate=LEARNING_RATE,\n",
                        "        logging_steps=25,\n",
                        "        optim='adamw_8bit',\n",
                        "        weight_decay=0.01,\n",
                        "        lr_scheduler_type='linear',\n",
                        "        seed=3407,\n",
                        "        fp16=not torch.cuda.is_bf16_supported(),\n",
                        "        bf16=torch.cuda.is_bf16_supported(),\n",
                        "        max_seq_length=MAX_SEQ_LENGTH,\n",
                        "        dataset_text_field='text',\n",
                        "        packing=False,\n",
                        "    ),\n",
                        ")\n",
                        "print('Trainer created!')"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Cell 8: Train!\n",
                        "print('🚀 Starting training...')\n",
                        "print('This may take a while depending on dataset size and epochs.')\n",
                        "print('-' * 50)\n",
                        "trainer_stats = trainer.train()\n",
                        "print('-' * 50)\n",
                        "print('✅ Training complete!')"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Cell 9: Save Model\n",
                        "print(f'Saving model to {OUTPUT_DIR}...')\n",
                        "model.save_pretrained(OUTPUT_DIR)\n",
                        "tokenizer.save_pretrained(OUTPUT_DIR)\n",
                        "print(f'✅ Model saved to {OUTPUT_DIR}')\n",
                        "print('\\nYou can find the model in your work/models folder!')"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Cell 10: (Optional) Export to GGUF for Ollama\n",
                        "# Uncomment the lines below to export\n",
                        "\n",
                        "# print('Exporting to GGUF format...')\n",
                        "# model.save_pretrained_gguf(\n",
                        "#     OUTPUT_DIR + '_gguf',\n",
                        "#     tokenizer,\n",
                        "#     quantization_method='q4_k_m'\n",
                        "# )\n",
                        "# print('GGUF export complete!')"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Cell 11: Test Inference\n",
                        "print('Testing inference...')\n",
                        "FastLanguageModel.for_inference(model)\n",
                        "\n",
                        "test_prompt = alpaca_prompt.format(\n",
                        "    instruction='Hello! Can you introduce yourself?',\n",
                        "    input='',\n",
                        "    output=''\n",
                        ")\n",
                        "\n",
                        "inputs = tokenizer(test_prompt, return_tensors='pt').to('cuda')\n",
                        "outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)\n",
                        "print('\\nModel response:')\n",
                        "print(tokenizer.decode(outputs[0], skip_special_tokens=True))"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.10.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        # Save notebook to work directory
        work_dir = self.workspace_path / "work"
        work_dir.mkdir(exist_ok=True)
        
        notebook_name = f"modelforge_train_{datetime.now().strftime('%Y%m%d_%H%M')}.ipynb"
        notebook_path = work_dir / notebook_name
        
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook_content, f, indent=2)
        
        return str(notebook_path)
    
    def generate_training_script(self, dataset_path: str) -> str:
        """Generate a Python training script for automated training."""
        dataset_name = Path(dataset_path).name
        
        # Use custom model name if provided, otherwise generate timestamp
        if self.config.custom_model_name and self.config.custom_model_name.strip():
            # Sanitize model name for filesystem
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.config.custom_model_name.strip())
            output_dir = f"/workspace/work/models/{safe_name}"
        else:
            output_dir = f"/workspace/work/models/{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        # Determine if using QLoRA (4-bit) or LoRA (16-bit)
        use_4bit = self.config.use_qlora
        
        script_content = f'''#!/usr/bin/env python3
"""
ModelForge Fine-Tuning Script
Auto-generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Run this script inside the Unsloth Docker container:
    python /workspace/work/train_script.py
"""

import sys
import threading
import time
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# Configuration
MODEL_NAME = "{self.config.model_name}"
MAX_SEQ_LENGTH = {self.config.max_seq_length}
USE_QLORA = {use_4bit}  # True = 4-bit QLoRA, False = 16-bit LoRA
DATASET_PATH = "/workspace/work/data/{dataset_name}"
OUTPUT_DIR = "{output_dir}"

# LoRA Config
LORA_R = {self.config.lora_r}
LORA_ALPHA = {self.config.lora_alpha}
LORA_DROPOUT = {self.config.lora_dropout}

# Training Config
NUM_EPOCHS = {self.config.num_train_epochs}
BATCH_SIZE = {self.config.per_device_train_batch_size}
GRAD_ACCUM = {self.config.gradient_accumulation_steps}
LEARNING_RATE = {self.config.learning_rate}
WARMUP_STEPS = {self.config.warmup_steps}

# GPU Monitoring
gpu_monitor_running = True

def get_gpu_stats():
    """Get current GPU statistics."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            if len(parts) >= 5:
                return {{
                    'name': parts[0].strip(),
                    'memory_used': float(parts[1]),
                    'memory_total': float(parts[2]),
                    'utilization': int(parts[3]),
                    'temperature': int(parts[4])
                }}
    except:
        pass
    return None

def gpu_monitor_thread():
    """Background thread to periodically print GPU stats."""
    while gpu_monitor_running:
        stats = get_gpu_stats()
        if stats:
            mem_pct = (stats['memory_used'] / stats['memory_total']) * 100
            print(f"[GPU] {{stats['name']}} | Mem: {{stats['memory_used']:.0f}}/{{stats['memory_total']:.0f}} MB ({{mem_pct:.1f}}%) | Util: {{stats['utilization']}}% | Temp: {{stats['temperature']}}°C", flush=True)
        time.sleep(10)  # Print every 10 seconds

# Start GPU monitoring in background
monitor_thread = threading.Thread(target=gpu_monitor_thread, daemon=True)
monitor_thread.start()

print("=" * 70, flush=True)
print("🔥 ModelForge Fine-Tuning", flush=True)
print("=" * 70, flush=True)

# Initial GPU info
stats = get_gpu_stats()
if stats:
    print(f"\\n[GPU INFO] {{stats['name']}}", flush=True)
    print(f"[GPU INFO] Total VRAM: {{stats['memory_total']:.0f}} MB", flush=True)
    print(f"[GPU INFO] Initial Usage: {{stats['memory_used']:.0f}} MB", flush=True)
print("", flush=True)

qlora_mode = "QLoRA (4-bit)" if USE_QLORA else "LoRA (16-bit)"
print(f"[CONFIG] Model: {{MODEL_NAME}}", flush=True)
print(f"[CONFIG] Mode: {{qlora_mode}}", flush=True)
print(f"[CONFIG] Dataset: {{DATASET_PATH}}", flush=True)
print(f"[CONFIG] Epochs: {{NUM_EPOCHS}}, Batch Size: {{BATCH_SIZE}}, LR: {{LEARNING_RATE}}", flush=True)
print(f"[CONFIG] LoRA r={{LORA_R}}, alpha={{LORA_ALPHA}}", flush=True)
print(f"[CONFIG] Output: {{OUTPUT_DIR}}", flush=True)
print("", flush=True)

print("[STEP 1/6] Loading base model...", flush=True)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=USE_QLORA,
)
print("✓ Model loaded!", flush=True)

print("[STEP 2/6] Applying LoRA adapters...", flush=True)
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
print("✓ LoRA applied!", flush=True)

print("[STEP 3/6] Loading dataset...", flush=True)
dataset = load_dataset('json', data_files=DATASET_PATH, split='train')
print(f"✓ Dataset loaded: {{len(dataset)}} samples", flush=True)

print("[STEP 4/6] Formatting dataset...", flush=True)
# Alpaca format - consistent with inference
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{{instruction}}

### Input:
{{input}}

### Response:
{{output}}"""

EOS_TOKEN = tokenizer.eos_token

def format_prompts(examples):
    instructions = examples['instruction']
    inputs = examples.get('input', [''] * len(instructions))
    if inputs is None:
        inputs = [''] * len(instructions)
    outputs = examples['output']
    texts = []
    for inst, inp, out in zip(instructions, inputs, outputs):
        # Format with Alpaca prompt and add EOS token
        text = alpaca_prompt.format(
            instruction=inst,
            input=inp if inp else '',
            output=out
        ) + EOS_TOKEN
        texts.append(text)
    return {{'text': texts}}

dataset = dataset.map(format_prompts, batched=True)
print("✓ Dataset formatted!", flush=True)

# Show sample
print("\\n[SAMPLE] First training example:", flush=True)
sample = dataset[0]['text'][:500]
print(sample + "..." if len(dataset[0]['text']) > 500 else sample, flush=True)
print("", flush=True)

print("[STEP 5/6] Creating trainer...", flush=True)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=1,  # Log every step for real-time updates
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
        report_to="none",  # Disable wandb etc
    ),
)
print("✓ Trainer created!", flush=True)

print("", flush=True)
print("=" * 70, flush=True)
print("[STEP 6/6] 🚀 TRAINING STARTED", flush=True)
print("=" * 70, flush=True)
print("", flush=True)

trainer_stats = trainer.train()

# Stop GPU monitoring
gpu_monitor_running = False
time.sleep(1)

print("", flush=True)
print("=" * 70, flush=True)
print("✅ TRAINING COMPLETE!", flush=True)
print("=" * 70, flush=True)
print("", flush=True)

# Training stats
if hasattr(trainer_stats, 'metrics'):
    metrics = trainer_stats.metrics
    print(f"[STATS] Training Loss: {{metrics.get('train_loss', 'N/A')}}", flush=True)
    print(f"[STATS] Training Time: {{metrics.get('train_runtime', 0):.1f}} seconds", flush=True)
    print(f"[STATS] Samples/Second: {{metrics.get('train_samples_per_second', 0):.2f}}", flush=True)

print("", flush=True)
print("[SAVING] Saving model and tokenizer...", flush=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✓ Model saved to: {{OUTPUT_DIR}}", flush=True)

# Final GPU stats
stats = get_gpu_stats()
if stats:
    print(f"", flush=True)
    print(f"[GPU FINAL] Memory Used: {{stats['memory_used']:.0f}} MB", flush=True)

print("", flush=True)
print("🎉 Fine-tuning complete! Your model is ready.", flush=True)
print(f"📁 Find it at: {{OUTPUT_DIR}}", flush=True)
'''
        
        # Save script to work directory
        work_dir = self.workspace_path / "work"
        work_dir.mkdir(exist_ok=True)
        
        # Copy dataset to work/data directory so it's accessible in container
        import shutil
        data_dir = self.workspace_path / "data"
        work_data_dir = work_dir / "data"
        work_data_dir.mkdir(exist_ok=True)
        
        # Copy the specific dataset file
        src_file = data_dir / dataset_name
        dst_file = work_data_dir / dataset_name
        if src_file.exists():
            shutil.copy2(src_file, dst_file)
            print(f"Copied dataset to {dst_file}")
        else:
            print(f"Warning: Dataset file not found at {src_file}")
        
        script_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M')}.py"
        script_path = work_dir / script_name
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        return str(script_path)
    
    def run_training_in_container(self, script_name: str) -> Tuple[bool, str]:
        """Run the training script inside the container."""
        try:
            container_id = self.get_running_container()
            if not container_id:
                return False, "Container is not running"
            
            container_script_path = f"/workspace/work/{script_name}"
            
            result = subprocess.run(
                ["docker", "exec", container_id, "python", container_script_path],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=7200  # 2 hour timeout
            )
            
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr
        except subprocess.TimeoutExpired:
            return False, "Training timed out after 2 hours"
        except Exception as e:
            return False, f"Error running training: {str(e)}"
    
    def generate_inference_script(self, model_path: str, prompt: str) -> str:
        """Generate a script to run inference on a trained model."""
        # Convert Windows path to container path if needed
        if "work/models" in model_path or "work\\models" in model_path:
            model_name = Path(model_path).name
            container_model_path = f"/workspace/work/models/{model_name}"
        else:
            container_model_path = model_path
        
        # Escape the prompt for Python string
        escaped_prompt = prompt.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'").replace('\n', '\\n')
        
        script_content = f'''#!/usr/bin/env python3
"""
ModelForge Inference Script (Optimized with Unsloth 2x Speed)
Auto-generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""

import locale
import sys

# Fix UTF-8 encoding for streaming output
locale.getpreferredencoding = lambda: "UTF-8"
sys.stdout.reconfigure(encoding='utf-8')

from unsloth import FastLanguageModel
import torch

MODEL_PATH = "{container_model_path}"
MAX_SEQ_LENGTH = 2048

print("=" * 60, flush=True)
print("🤖 ModelForge Inference (Unsloth 2x Speed)", flush=True)
print("=" * 60, flush=True)
print("", flush=True)

print(f"[LOADING] Model from {{MODEL_PATH}}...", flush=True)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)

# Enable Unsloth's 2x faster inference
FastLanguageModel.for_inference(model)
print("✓ Model loaded with Unsloth 2x inference optimization!", flush=True)
print("", flush=True)

# Alpaca format prompt - MUST match training format exactly
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{{instruction}}

### Input:
{{input}}

### Response:
"""

user_instruction = """{escaped_prompt}"""

formatted_prompt = alpaca_prompt.format(
    instruction=user_instruction,
    input=""
)

print("=" * 60, flush=True)
print("📝 USER PROMPT:", flush=True)
print(user_instruction, flush=True)
print("=" * 60, flush=True)
print("", flush=True)
print("🤖 MODEL RESPONSE:", flush=True)
print("-" * 60, flush=True)

inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

# Use TextStreamer for streaming output with Unsloth optimization
from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# Generate with optimized settings for speed
_ = model.generate(
    **inputs,
    streamer=text_streamer,
    max_new_tokens=512,
    use_cache=True,  # Required for Unsloth 2x speed
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
)

print("", flush=True)
print("-" * 60, flush=True)
print("✅ Inference complete!", flush=True)
'''
        
        # Save script to work directory
        work_dir = self.workspace_path / "work"
        work_dir.mkdir(exist_ok=True)
        
        script_path = work_dir / "inference_script.py"
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        return str(script_path)
    
    def run_inference_in_container(self, model_path: str, prompt: str) -> Tuple[bool, str]:
        """Run inference on a trained model inside the container."""
        try:
            container_id = self.get_running_container()
            if not container_id:
                return False, "Container is not running"
            
            # Generate the inference script
            script_path = self.generate_inference_script(model_path, prompt)
            script_name = Path(script_path).name
            container_script_path = f"/workspace/work/{script_name}"
            
            # Run with real-time output capture
            result = subprocess.run(
                ["docker", "exec", container_id, "python", "-u", container_script_path],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                return True, result.stdout if result.stdout else "No output received"
            else:
                error_msg = result.stderr if result.stderr else "Unknown error"
                return False, f"Inference error: {error_msg}"
        except subprocess.TimeoutExpired:
            return False, "Inference timed out after 5 minutes"
        except Exception as e:
            return False, f"Error running inference: {str(e)}"


# Legacy alias for compatibility
FineTuningManager = DockerFineTuningManager


@dataclass
class TrainingStatus:
    """Tracks the status of an ongoing training run."""
    is_running: bool = False
    is_complete: bool = False
    has_error: bool = False
    error_message: str = ""
    output_lines: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    model_output_dir: str = ""
    # GPU stats
    gpu_name: str = ""
    gpu_memory_total: float = 0.0
    gpu_memory_used: float = 0.0
    gpu_utilization: int = 0
    gpu_temperature: int = 0
    
    def elapsed_time(self) -> str:
        """Get formatted elapsed time."""
        if not self.start_time:
            return "0:00"
        end = self.end_time or datetime.now()
        elapsed = end - self.start_time
        minutes, seconds = divmod(int(elapsed.total_seconds()), 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"


# Global training status - persists across Streamlit reruns
_training_status: Optional[TrainingStatus] = None
_training_thread: Optional[threading.Thread] = None


def get_training_status() -> Optional[TrainingStatus]:
    """Get the current training status."""
    global _training_status
    return _training_status


def reset_training_status():
    """Reset the training status."""
    global _training_status, _training_thread
    _training_status = None
    _training_thread = None


def start_training_async(
    manager: DockerFineTuningManager,
    dataset_path: str,
    on_complete: Optional[Callable] = None
) -> TrainingStatus:
    """Start training in a background thread with real-time output."""
    global _training_status, _training_thread
    
    # Check if already running
    if _training_status and _training_status.is_running:
        return _training_status
    
    # Determine output directory name
    if manager.config.custom_model_name and manager.config.custom_model_name.strip():
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in manager.config.custom_model_name.strip())
        output_dir_name = safe_name
    else:
        output_dir_name = datetime.now().strftime('%Y%m%d_%H%M')
    
    # Create fresh status
    _training_status = TrainingStatus(
        is_running=True,
        start_time=datetime.now(),
        model_output_dir=f"work/models/{output_dir_name}"
    )
    
    def parse_gpu_stats(line: str):
        """Parse GPU stats from log line."""
        global _training_status
        if "[GPU]" in line or "[GPU INFO]" in line:
            try:
                # Parse GPU info line
                if "Total VRAM:" in line:
                    import re
                    match = re.search(r'(\d+(?:\.\d+)?)\s*MB', line)
                    if match:
                        _training_status.gpu_memory_total = float(match.group(1))
                elif "[GPU]" in line and "|" in line:
                    # Format: [GPU] NAME | Mem: USED/TOTAL MB (PCT%) | Util: X% | Temp: Y°C
                    parts = line.split("|")
                    if len(parts) >= 4:
                        # Get GPU name
                        name_part = parts[0].replace("[GPU]", "").strip()
                        _training_status.gpu_name = name_part
                        
                        # Get memory
                        import re
                        mem_match = re.search(r'(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)\s*MB', parts[1])
                        if mem_match:
                            _training_status.gpu_memory_used = float(mem_match.group(1))
                            _training_status.gpu_memory_total = float(mem_match.group(2))
                        
                        # Get utilization
                        util_match = re.search(r'Util:\s*(\d+)%', parts[2])
                        if util_match:
                            _training_status.gpu_utilization = int(util_match.group(1))
                        
                        # Get temperature
                        temp_match = re.search(r'Temp:\s*(\d+)', parts[3])
                        if temp_match:
                            _training_status.gpu_temperature = int(temp_match.group(1))
            except:
                pass
    
    def run_training():
        global _training_status
        try:
            # Generate script
            script_path = manager.generate_training_script(dataset_path)
            script_name = Path(script_path).name
            
            _training_status.output_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] 📝 Generated training script: {script_name}")
            _training_status.output_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🐳 Starting training in Docker container...")
            _training_status.output_lines.append("")
            
            # Get container ID
            container_id = manager.get_running_container()
            if not container_id:
                raise Exception("Container is not running. Please start the container first.")
            
            container_script_path = f"/workspace/work/{script_name}"
            
            # Run with real-time output (with UTF-8 encoding for Windows compatibility)
            process = subprocess.Popen(
                ["docker", "exec", container_id, "python", "-u", container_script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1
            )
            
            # Stream output
            for line in iter(process.stdout.readline, ''):
                if line:
                    clean_line = line.rstrip()
                    _training_status.output_lines.append(clean_line)
                    
                    # Parse GPU stats
                    parse_gpu_stats(clean_line)
                    
                    # Keep only last 300 lines to avoid memory issues
                    if len(_training_status.output_lines) > 300:
                        _training_status.output_lines = _training_status.output_lines[-300:]
            
            process.wait()
            
            if process.returncode == 0:
                _training_status.is_complete = True
                _training_status.output_lines.append("")
                _training_status.output_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Training completed successfully!")
            else:
                raise Exception(f"Training failed with exit code {process.returncode}")
                
        except Exception as e:
            _training_status.has_error = True
            _training_status.error_message = str(e)
            _training_status.output_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Error: {str(e)}")
        finally:
            _training_status.is_running = False
            _training_status.end_time = datetime.now()
            if on_complete:
                on_complete()
    
    # Start background thread
    _training_thread = threading.Thread(target=run_training, daemon=True)
    _training_thread.start()
    
    return _training_status
