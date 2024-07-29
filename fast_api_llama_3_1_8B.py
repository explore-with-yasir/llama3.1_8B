from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, Body, Query, Request
from typing import List
from huggingface_hub import hf_hub_download, hf_hub_url
import uvicorn
import torch

# Import necessary components from transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

# Create a FastAPI application instance
app = FastAPI()

# Hugging Face Access Token (replace with your own)
access_token = ''

# Model ID from Hugging Face Hub
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Load tokenizer and model from Hugging Face Hub (requires access token)
tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
model = AutoModelForCausalLM.from_pretrained(model_id, token=access_token)

# Move the model to GPU if available, otherwise CPU
if torch.cuda.is_available():
    model = model.to("cuda")
else:
    model = model.to("cpu")

# Define conversation termination tokens
terminators = [
    tokenizer.eos_token_id,  # End-of-sentence token
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),  # Custom end-of-conversation token
]

# Maximum allowed input token length
MAX_INPUT_TOKEN_LENGTH = 4096

# Endpoint for text generation
@app.post("/generate")
async def generate(request: Request):
    # Parse request body as JSON
    data = await request.json()

    # Validate required field: message
    message = data.get("message")
    if not message:
        return {"error": "Missing message"}

    # Extract optional parameters (defaults provided)
    history = data.get("history", [])  # List of conversation turns
    temperature = data.get("temperature", 0.7)  # Controls randomness (0 = greedy, 1 = more random)
    max_new_tokens = data.get("max_new_tokens", 256)  # Maximum number of tokens to generate
    system = data.get("system", "")  # Optional system prompt (if any)

    # Construct conversation history with roles
    conversation = []
    if system:
        conversation.append({"role": "system", "content": system})

    for user, assistant in zip(*history):
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})   


    # Convert conversation to token IDs
    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")

    # Truncate input if exceeding maximum length
    if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
        input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]

    # Move input to model's device (GPU or CPU)
    input_ids = input_ids.to(model.device)

    # Generate text with specified parameters
    generate_kwargs = {
        "input_ids": input_ids,
        "max_length": max_new_tokens + input_ids.shape[1],  # Adjust for total length
        "do_sample": temperature != 0,  # Use sampling for non-zero temperature (randomness)
        "temperature": temperature,
        "eos_token_id": terminators,  # Specify tokens to stop generation
    }

    # Generate output tokens and decode them to text
    output = model.generate(**generate_kwargs)[0]
    response = tokenizer.decode(output, skip_special_tokens=True)

    # Return the generated response
    return {"response": response}

# Run the FastAPI application if executed directly (for development purposes)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
