# Experiment 1 - Natural Language to Temporal Logic

This experiment evaluates large language models (LLMs) on the task of translating natural language instructions into temporal logic (TL) specifications with BNF grammar constraints.

## Setup

### 1. Setup llama-cpp-python backend (optional)

Follow [llama-cpp-python README instructions](https://github.com/abetlen/llama-cpp-python) to set environment variables to build llama-cpp-python for backends like CUDA. Our trials were run on a desktop computer with a 24-core i9 CPU, 32 GB RAM, and an NVIDIA RTX 4090 GPU with CUDA.

### 2. Install required Python packages

```
pip install -r requirements.txt
```

### 3. Get access to Gemma3 on HuggingFace

Gemma 3 is a gated model. You must accept Google's usage agreement in the [model card](https://huggingface.co/google/gemma-3-27b-it-qat-q4_0-gguf) and login to huggingface-cli with your token to download the model.

```
huggingface-cli login
```
### 4. Add OpenAI API key environment variable

```
export OPENAI_API_KEY='yourkey'
```

This step is only required to compare our method's performance using Gemma 3 against OpenAI models like GPT-4. If you do not want to use OpenAI credits, you can comment out this ablation, as described below.

### 5. Run experiment

```
usage: run_experiment.py [-h] --entries ENTRIES --trials TRIALS --examples EXAMPLES --dataset_jsonl DATASET_JSONL

Experiment configuration

options:
  -h, --help            show this help message and exit
  --entries ENTRIES     Number of dataset entries to use
  --trials TRIALS       Number of trials to run
  --examples EXAMPLES   Number of few-shot examples to include
  --dataset_jsonl DATASET_JSONL
                        Path to the input JSONL dataset (e.g., navi_total_refined.jsonl)
```

You may edit the `ablations` variable in the main function of `run_experiment.py` to enable or disable certain ablations described in our paper. This may be useful if you would like to run our code without using OpenAI API credits.

