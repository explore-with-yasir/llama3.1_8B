from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, Body, Query, Request
from typing import List
from huggingface_hub import hf_hub_download, hf_hub_url
import uvicorn
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

access_token = '' #Hugging Face Access Token to download the model
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
model = AutoModelForCausalLM.from_pretrained(model_id, token=access_token)
if torch.cuda.is_available():
    model = model.to("cuda")  # Move the model to GPU
else:
    model = model.to("cpu")  # Move the model to CPU (default)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]
MAX_INPUT_TOKEN_LENGTH = 4096

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    message = data.get("message")
    if not message:
        return {"error": "Missing message"}
    history = data.get("history")
    temperature = data.get("temperature")
    max_new_tokens = data.get("max_new_tokens")
    system = data.get("system")
    conversation = []
    if system:
        conversation.append({"role": "system", "content": system})

    for user, assistant in zip(*history):
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})
    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")
    if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
        input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]

    input_ids = input_ids.to(model.device)

    generate_kwargs = {
        "input_ids": input_ids,
        "max_length": max_new_tokens + input_ids.shape[1],  # Adjust for total length
        "do_sample": temperature != 0,  # Greedy generation for temp=0
        "temperature": temperature,
        "eos_token_id": terminators,
    }

    output = model.generate(**generate_kwargs)[0]
    response = tokenizer.decode(output, skip_special_tokens=True)

    return {"response": response}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
