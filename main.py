#!/usr/bin/env python3
"""
Directory structure expected:
examples/
├── example1.jpg
├── example1.txt
├── example2.png
├── example2.txt
└── ...

Usage:
python main.py --examples-dir ./examples --image ./target_image.jpg
"""

import os
import argparse
import base64
import json
from pathlib import Path
from typing import List, Dict, Tuple
import openai
from openai import OpenAI

SYSTEM_PROMPT = "You are an intelligent AI model onboard a warehouse mobile robot without manipulation capabilities, equipped with a camera. You will be called when the robot encounters an unknown obstacle and tasked with thinking deeply about the situation."
USER_PROMPT = "The robot has encountered an unidentified obstacle at location (3,5) and captured this image from its forward facing camera. Generate a detailed description of the scene which can be used to understand why the robot cannot proceed. Then, using this scene description, generate a detailed help request that will be broadcast to all other robots in the warehouse. Think deeply about your own capabilities and what capabilities are needed to resolve this obstacle. Include all information necessary for another robot to help you, including your location."
HELP_REQ_PROMPT = ""

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_example_pairs(examples_dir: str) -> List[Tuple[str, str]]:
    examples_path = Path(examples_dir)
    if not examples_path.exists():
        raise FileNotFoundError(f"Examples directory not found: {examples_dir}")
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    
    examples = []
    
    for image_file in examples_path.iterdir():
        if image_file.suffix.lower() in image_extensions:
            text_file = image_file.with_suffix('.txt')
            if text_file.exists():
                try:
                    with open(text_file, 'r', encoding='utf-8') as f:
                        text_content = f.read().strip()
                    examples.append((str(image_file), text_content))
                    print(f"Loaded example: {image_file.name} -> {text_file.name}")
                except Exception as e:
                    print(f"Error reading {text_file}: {e}")
            else:
                print(f"Warning: No corresponding text file found for {image_file.name}")
    
    if not examples:
        print("Warning: No valid image-text pairs found in examples directory")
    
    return examples

def create_few_shot_messages(examples: List[Tuple[str, str]], base_prompt: str) -> List[Dict]:
    messages = []
    
    messages.append({
        "role": "system",
        "content": SYSTEM_PROMPT
    })
    
    for i, (image_path, text_content) in enumerate(examples):
        try:
            image_base64 = encode_image_to_base64(image_path)
            image_url = f"data:image/jpeg;base64,{image_base64}"
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": base_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            })
            
            messages.append({
                "role": "assistant",
                "content": text_content
            })
            
        except Exception as e:
            print(f"Error processing example {image_path}: {e}")
            continue
    
    return messages

def call_openai_vlm(client: OpenAI, messages: List[Dict], model: str = "gpt-4o") -> str:
    """
    Call OpenAI's Vision Language Model.
    
    Args:
        client: OpenAI client instance
        messages: List of messages for the API call
        model: Model name to use
        
    Returns:
        Generated response text
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1000,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error calling OpenAI API: {e}")

def main():
    parser = argparse.ArgumentParser(description="Warehouse Conflict VLM")
    parser.add_argument("--examples-dir", required=True, help="Directory containing example image-text pairs")
    parser.add_argument("--image", required=True, help="Path to target image to analyze")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use (default: gpt-4o)")
    parser.add_argument("--output", help="Output file to save result (optional)")
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not provided. Use --api-key or set OPENAI_API_KEY environment variable.")
    
    client = OpenAI(api_key=api_key)
    
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Target image not found: {args.image}")
    
    print(f"Loading examples from: {args.examples_dir}")
    examples = load_example_pairs(args.examples_dir)
    print(f"Found {len(examples)} example pairs")
    
    messages = create_few_shot_messages(examples, USER_PROMPT)
    
    try:
        target_image_base64 = encode_image_to_base64(args.image)
        target_image_url = f"data:image/jpeg;base64,{target_image_base64}"
        
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": USER_PROMPT},
                {"type": "image_url", "image_url": {"url": target_image_url}}
            ]
        })
        
    except Exception as e:
        raise Exception(f"Error processing target image {args.image}: {e}")
    
    print(f"Calling OpenAI {args.model} with {len(examples)} few-shot examples...")
    
    try:
        result = call_openai_vlm(client, messages, args.model)
        
        print("\n" + "="*50)
        print("RESULT:")
        print("="*50)
        print(result)
        print("="*50)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"\nResult saved to: {args.output}")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())