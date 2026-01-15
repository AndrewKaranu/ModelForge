# 🔥 ModelForge

**Synthetic Data Generation & Model Fine-Tuning Platform**

ModelForge is an end-to-end platform for creating synthetic training data and fine-tuning LLMs using the Backboard.io SDK and Unsloth.

![ModelForge](logo.png)

---

## Features

### Synthetic Data Generation
Generate high-quality training data with 5 specialized modes:

| Mode | Description |
|------|-------------|
| **General Chat** | Instruction/response pairs on any topic |
| **RAG/Document** | Q&A based on uploaded documents or URLs |
| **Code Generation** | Programming tasks with solutions |
| **Agent/Tool Use** | Function calling and tool usage examples |
| **Chain of Thought** | Step-by-step reasoning demonstrations |

###  Model Fine-Tuning
- **Docker-based Unsloth** training (works on Windows!)
- **QLoRA** support for efficient training on consumer GPUs
- Multiple base models: Llama, Qwen, Phi, Gemma
- Automatic Alpaca format conversion

### Model Arena
- Compare fine-tuned models against base models
- Optional AI judge for automated evaluation
- Side-by-side output comparison

### 2200+ LLM Models
- Direct access to any model from [Backboard Model Library](https://app.backboard.io/dashboard/model-library)
- Simple `provider/model-name` format (e.g., `openai/gpt-4o`)

---

## Quick Start

### 1. Prerequisites
- Python 3.10+
- Docker Desktop (for fine-tuning)
- NVIDIA GPU with CUDA (recommended)

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/AndrewKaranu/ModelForge.git
cd ModelForge

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### 3. Configuration

Edit `.env` and add your Backboard API key:
```
BACKBOARD_API_KEY=your_api_key_here
```

Get your free API key at [app.backboard.io](https://app.backboard.io)

### 4. Run the App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Usage Guide

### Step 1: Generate Training Data

1. Enter your Backboard API key in the sidebar
2. Click **"+ New Pipeline"** to create a data generation pipeline
3. Select a generation mode (General, RAG, Code, Agent, or Chain of Thought)
4. Configure the model and parameters
5. Click **"Generate"** to create synthetic data
6. Export as JSONL for training

### Step 2: Fine-Tune a Model

1. Go to the **Fine-Tuning** page
2. Select a base model (e.g., `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`)
3. Choose your generated dataset
4. Configure training parameters (epochs, learning rate, etc.)
5. Click **"Start Training"** to begin fine-tuning in Docker

### Step 3: Test Your Model

1. Go to the **Inference** page
2. Select your fine-tuned model
3. Enter a test prompt
4. Compare outputs in **Arena Mode** against base models

---

## Tech Stack

- **Frontend**: Streamlit
- **LLM Routing**: [Backboard.io SDK](https://backboard.io) (2200+ models)
- **Fine-Tuning**: [Unsloth](https://github.com/unslothai/unsloth) via Docker
- **Training**: QLoRA with 4-bit quantization

---


## Model Format

When selecting models, use the format: `provider/model-name`

**Examples:**
- `openai/gpt-4o`
- `anthropic/claude-3-5-sonnet-20241022`
- `google/gemini-2.0-flash-exp`
- `deepseek/deepseek-chat`
- `qwen/qwen-2.5-coder-32b-instruct`

Browse all models at the [Backboard Model Library](https://app.backboard.io/dashboard/model-library).

---

## 📝 License

MIT License

---


