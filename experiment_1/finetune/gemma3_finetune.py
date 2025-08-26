from unsloth import FastModel
import torch

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3-4b-it",
    max_seq_length = 2048, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # SHould leave on always!

    r = 8,           # Larger = higher accuracy, but might overfit
    lora_alpha = 8,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)

from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)

from datasets import load_dataset
# dataset = load_dataset("mlabonne/FineTome-100k", split = "train")
dataset = load_dataset('json', data_files='sharegpt_dataset.jsonl')

dataset = dataset['train'].train_test_split(
    test_size = 0.01, # 1% for test size can also be an integer for # of rows
    shuffle = True, # Should always set to True!
    seed = 3407,
)
train_dataset = dataset['train']
test_dataset = dataset['test']

from unsloth.chat_templates import standardize_data_formats
train_dataset = standardize_data_formats(train_dataset)
test_dataset = standardize_data_formats(test_dataset)

train_dataset[100]

def formatting_prompts_func(examples):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
   return { "text" : texts, }

train_dataset = train_dataset.map(formatting_prompts_func, batched = True)
test_dataset = test_dataset.map(formatting_prompts_func, batched = True)

train_dataset[100]["text"]

from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = test_dataset, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 30,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use this for WandB etc
        fp16_full_eval = True,
        per_device_eval_batch_size = 2,
        eval_accumulation_steps = 4,
        eval_strategy = "steps",
        eval_steps = 10,
    ),
)

from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)

tokenizer.decode(trainer.train_dataset[100]["input_ids"])

tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, " ")

# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# from unsloth.chat_templates import get_chat_template
#tokenizer = get_chat_template(
#    tokenizer,
#    chat_template = "gemma-3",
#)
#messages = [{
#    "role": "user",
#    "content": [{
#        "type" : "text",
#        "text" : "Natural language task - Go to the kitchen first, then go to the living room and wait there until the light turns on.",
#    }]
#}]
#text = tokenizer.apply_chat_template(
#    messages,
#    add_generation_prompt = True, # Must add for generation
#)
#outputs = model.generate(
#    **tokenizer([text], return_tensors = "pt").to("cuda"),
#    max_new_tokens = 64, # Increase for longer outputs!
#    # Recommended Gemma-3 settings!
#    temperature = 1.0, top_p = 0.95, top_k = 64,
#)
#tokenizer.batch_decode(outputs)

#messages = [{
#    "role": "user",
#    "content": [{"type" : "text", "text" : "Why is the sky blue?",}]
#}]
#text = tokenizer.apply_chat_template(
#    messages,
#    add_generation_prompt = True, # Must add for generation
#)

#from transformers import TextStreamer
#_ = model.generate(
#    **tokenizer([text], return_tensors = "pt").to("cuda"),
#    max_new_tokens = 64, # Increase for longer outputs!
#    # Recommended Gemma-3 settings!
#    temperature = 1.0, top_p = 0.95, top_k = 64,
#    streamer = TextStreamer(tokenizer, skip_prompt = True),
#)

#model.save_pretrained("gemma-3")  # Local saving
#tokenizer.save_pretrained("gemma-3")
# model.push_to_hub("HF_ACCOUNT/gemma-3", token = "...") # Online saving
# tokenizer.push_to_hub("HF_ACCOUNT/gemma-3", token = "...") # Online saving

if False:
    from unsloth import FastModel
    model, tokenizer = FastModel.from_pretrained(
        model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = 2048,
        load_in_4bit = True,
    )

#messages = [{
#    "role": "user",
#    "content": [{"type" : "text", "text" : "What is Gemma-3?",}]
#}]
#text = tokenizer.apply_chat_template(
#    messages,
#    add_generation_prompt = True, # Must add for generation
#)

#from transformers import TextStreamer
#_ = model.generate(
#    **tokenizer([text], return_tensors = "pt").to("cuda"),
#    max_new_tokens = 64, # Increase for longer outputs!
#    # Recommended Gemma-3 settings!
#    temperature = 1.0, top_p = 0.95, top_k = 64,
#    streamer = TextStreamer(tokenizer, skip_prompt = True),
#)

if True: # Change to True to save finetune!
    model.save_pretrained_merged("gemma-3-finetune", tokenizer)

if False: # Change to True to upload finetune
    model.push_to_hub_merged(
        "HF_ACCOUNT/gemma-3-finetune", tokenizer,
        token = "hf_..."
    )

if False: # Change to True to save to GGUF
    model.save_pretrained_merged("gemma-3-finetune", tokenizer)
    model.save_pretrained_gguf(
        "gemma-3-finetune",
        tokenizer,
        quantization_type = "Q8_0", # For now only Q8_0, BF16, F16 supported
    )

if False: # Change to True to upload GGUF
    model.push_to_hub_gguf(
        "gemma-3-finetune",
        quantization_type = "Q8_0", # Only Q8_0, BF16, F16 supported
        repo_id = "HF_ACCOUNT/gemma-finetune-gguf",
        token = "hf_...",
    )

