import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
import torch
import torch.nn.functional as F
from nnsight import LanguageModel
from typing import List
import pandas as pd

# Set up the API key for nnsight
from nnsight import CONFIG
import os
api_key = os.getenv('NNSIGHT_API_KEY')
CONFIG.set_default_api_key(api_key)

# Load the Language Model
llama = LanguageModel("meta-llama/Meta-Llama-3.1-8B", device="cuda")

def run_lens(model,PROMPT):
    logits_lens_token_result_by_layer = []
    logits_lens_probs_by_layer = []
    logits_lens_ranks_by_layer = []
    input_ids = model.tokenizer.encode(PROMPT)
    with model.trace(input_ids, remote=True) as runner:
        for layer_ix,layer in enumerate(model.model.layers):
            hidden_state = layer.output[0][0]
            logits_lens_normed_last_token = model.model.norm(hidden_state)
            logits_lens_token_distribution = model.lm_head(logits_lens_normed_last_token)
            logits_lens_last_token_logits = logits_lens_token_distribution[-1:]
            logits_lens_probs = F.softmax(logits_lens_last_token_logits, dim=1).save()
            logits_lens_probs_by_layer.append(logits_lens_probs)
            logits_lens_next_token = torch.argmax(logits_lens_probs, dim=1).save()
            logits_lens_token_result_by_layer.append(logits_lens_next_token)
        tokens_out = llama.lm_head.output.argmax(dim=-1).save()
        expected_token = tokens_out[0][-1].save()
    logits_lens_all_probs = np.concatenate([probs[:, expected_token].cpu().detach().numpy() for probs in logits_lens_probs_by_layer])
    #get the rank of the expected token from each layer's distribution
    for layer_probs in logits_lens_probs_by_layer:
        # Sort the probabilities in descending order and find the rank of the expected token
        sorted_probs, sorted_indices = torch.sort(layer_probs, descending=True)
        # Find the rank of the expected token (1-based rank)
        expected_token_rank = (sorted_indices == expected_token).nonzero(as_tuple=True)[1].item() + 1
        logits_lens_ranks_by_layer.append(expected_token_rank)
    actual_output = llama.tokenizer.decode(expected_token.item())
    logits_lens_results = [model.tokenizer.decode(next_token.item()) for next_token in logits_lens_token_result_by_layer]
    return logits_lens_results, logits_lens_all_probs, actual_output,logits_lens_ranks_by_layer

def plot_prob(PROMPT, logits_lens_results, logits_lens_all_probs, actual_output):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(logits_lens_all_probs)), logits_lens_all_probs, marker='x')

    for layer_number, next_token in enumerate(logits_lens_results):
        plt.text(layer_number, logits_lens_all_probs[layer_number], next_token, ha='right', va='top', fontsize=8)

    plt.xlabel('Layer Number')
    plt.ylabel(f'Probability of Expected Token "{actual_output}"')
    plt.title(f'Logits Lens for "{PROMPT}"')
    plt.grid(True)
    plt.ylim(0.0, 1.0)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    return img

def plot_rank(PROMPT,actual_output,logits_lens_ranks_by_layer):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(logits_lens_ranks_by_layer)), logits_lens_ranks_by_layer, marker='x')
    for layer_number, rank in enumerate(logits_lens_ranks_by_layer):
        plt.text(layer_number, rank, rank, ha='right', va='top',fontsize=8)
    plt.xlabel('Layer Number')
    plt.ylabel(f'Rank of Expected Token "{actual_output}"')
    plt.title(f'Logits Lens for "{PROMPT}"')
    plt.grid(True)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    return img

def process_file(file_path):
    """Read uploaded file and return list of prompts."""
    prompts = []
    if file_path is None:
        return prompts
    
    if file_path.endswith('.csv'):
        # Process CSV file
        df = pd.read_csv(file_path)
        if 'Prompt' in df.columns:
            prompts = df[['Prompt']].dropna().values.tolist()
    
    # Read the file as text and split into lines (one prompt per line)
    prompts = []
    with open(file_path, 'r') as file:
        prompts = [[line] for line in file.read().splitlines()]

    return prompts

def gradio_interface(prompts: List[str], prompt_file=None):
    results = []
    prob_graphs = []
    rank_graphs = []
    
    # Process the file if uploaded
    if prompt_file is not None:
        file_prompts = process_file(prompt_file)
        prompts.extend(file_prompts)

    # Ensure to process each prompt in the list
    for prompt in prompts:
        # If a prompt is an empty string, skip it
        prompt = prompt[0]
        if not prompt:
            continue
        lens_output = run_lens(llama, prompt)
        p_graph = plot_prob(prompt, lens_output[0], lens_output[1], lens_output[2])
        r_graph = plot_rank(prompt, lens_output[2], lens_output[3])
        results.append(p_graph)
        results.append(r_graph) 
        prob_graphs.append(p_graph)
        rank_graphs.append(r_graph)

    return results

# Define Gradio interface with both DataFrame input and file upload
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Dataframe(headers=["Prompt"], row_count=5, col_count=1, type="array"),  # Table input
        gr.File(type="filepath", label="Upload a File with Prompts")  # File input
    ],
    outputs=gr.Gallery(type="pil", label="Visualizations"),
    title="Logit Lens Visualization",
    description="Upload a table of prompts or a file to visualize how token probabilities change across layers.",
    allow_flagging=False
)

# Launch the Gradio app
iface.launch()