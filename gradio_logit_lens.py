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
from adjustText import adjust_text

# Set up the API key for nnsight
from nnsight import CONFIG
import os
api_key = os.getenv('NNSIGHT_API_KEY')
CONFIG.set_default_api_key(api_key)


# Load the Language Model
# llama = LanguageModel("meta-llama/Meta-Llama-3.1-8B")

# Model options
MODEL_OPTIONS = {
    "Llama3.1-8B": "meta-llama/Meta-Llama-3.1-8B",
    "Llama3.1-70B": "meta-llama/Meta-Llama-3.1-70B",
}


#placeholder for reset
prompts_with_probs = pd.DataFrame(
{
    "prompt": [''],
    "layer": [0],
    "results": [''],
    "probs": [0],
    "expected": [''],
})
prompts_with_ranks = pd.DataFrame(
{
    "prompt": [''],
    "layer": [0],
    "results": [''],
    "ranks": [0],
    "expected": [''],
})

def run_lens(model,PROMPT):
    logit_lens_token_result_by_layer = []
    logit_lens_probs_by_layer = []
    logit_lens_ranks_by_layer = []
    input_ids = model.tokenizer.encode(PROMPT)
    with model.trace(input_ids, remote=True) as runner:
        for layer_ix,layer in enumerate(model.model.layers):
            hidden_state = layer.output[0][0]
            logit_lens_normed_last_token = model.model.norm(hidden_state)
            logit_lens_token_distribution = model.lm_head(logit_lens_normed_last_token)
            logit_lens_last_token_logit = logit_lens_token_distribution[-1:]
            logit_lens_probs = F.softmax(logit_lens_last_token_logit, dim=1).save()
            logit_lens_probs_by_layer.append(logit_lens_probs)
            logit_lens_next_token = torch.argmax(logit_lens_probs, dim=1).save()
            logit_lens_token_result_by_layer.append(logit_lens_next_token)
        tokens_out = model.lm_head.output.argmax(dim=-1).save()
        expected_token = tokens_out[0][-1].save()
    logit_lens_all_probs = np.concatenate([probs[:, expected_token].cpu().detach().to(torch.float32).numpy() for probs in logit_lens_probs_by_layer])

    #get the rank of the expected token from each layer's distribution
    for layer_probs in logit_lens_probs_by_layer:
        # Sort the probabilities in descending order and find the rank of the expected token
        sorted_probs, sorted_indices = torch.sort(layer_probs, descending=True)
        # Find the rank of the expected token (1-based rank)
        expected_token_rank = (sorted_indices == expected_token).nonzero(as_tuple=True)[1].item() + 1
        logit_lens_ranks_by_layer.append(expected_token_rank)
    actual_output = model.tokenizer.decode(expected_token.item())
    logit_lens_results = [model.tokenizer.decode(next_token.item()) for next_token in logit_lens_token_result_by_layer]
    return logit_lens_results, logit_lens_all_probs, actual_output,logit_lens_ranks_by_layer


def process_file(prompts_data,file_path):
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
    else:
        with open(file_path, 'r') as file:
            prompts = [[line] for line in file.read().splitlines()]

    for prompt in prompts_data:
        if prompt==['']:
            continue
        else:
            prompts.append(prompt)

    return prompts

def plot_prob(prompts_with_probs):
    plt.figure(figsize=(10, 6))
    texts = []  # List to hold text annotations for adjustment

    # Iterate over each prompt and plot its probabilities
    for prompt in prompts_with_probs['prompt'].unique():
        # Filter the DataFrame for the current prompt
        prompt_data = prompts_with_probs[prompts_with_probs['prompt'] == prompt]
        label = f"{prompt}({prompt_data['expected'].iloc[0]})"
        
        # Plot probabilities for this prompt
        plt.plot(prompt_data['layer'], prompt_data['probs'], marker='x', label=label)

        # Annotate each point with the corresponding result
        for layer, prob, result in zip(prompt_data['layer'], prompt_data['probs'], prompt_data['results']):
            text = plt.text(layer, prob, result, fontsize=8)
            texts.append(text)

    # Add labels and title
    plt.xlabel('Layer Number')
    plt.ylabel('Probability')
    plt.title('Probability of most-likely output token')
    plt.grid(True)
    plt.xlim(0,max(prompts_with_probs['layer']))
    plt.ylim(0.0, 1.0)
    plt.legend(title='Prompts', bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=1)

    # Adjust text to prevent overlap
    adjust_text(texts, only_move={'points': 'xy', 'texts': 'xy'}, 
                arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

    # Save the plot
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    return img

def plot_rank(prompts_with_ranks):
    plt.figure(figsize=(10, 6))
    texts = []

    # Iterate over each prompt and plot its ranks
    for prompt in prompts_with_ranks['prompt'].unique():
        # Filter the DataFrame for the current prompt
        prompt_data = prompts_with_ranks[prompts_with_ranks['prompt'] == prompt]
        label = f"{prompt}({prompt_data['expected'].iloc[0]})"
        
        # Plot ranks for this prompt
        plt.plot(prompt_data['layer'], prompt_data['ranks'], marker='x', label=label)

        # Annotate each point with the corresponding result
        for layer, rank, result in zip(prompt_data['layer'], prompt_data['ranks'], prompt_data['results']):
            text = plt.text(layer, rank, result, ha='right', va='bottom', fontsize=8)
            texts.append(text)

    # Add labels and title
    plt.xlabel('Layer Number')
    plt.ylabel('Rank')
    plt.title('Rank of most-likely output token')
    plt.grid(True)
    plt.xlim(0,max(prompts_with_ranks['layer']))
    plt.ylim(bottom=0)
    plt.legend(title='Prompts', bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=1)

    # Adjust text to prevent overlap
    adjust_text(texts,only_move={'points': 'xy', 'texts': 'xy'},
                arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

    # Save the plot
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    return img

def submit_prompts(model_name, prompts_data):
    llama = LanguageModel(MODEL_OPTIONS[model_name])
    # Initialize lists to accumulate results
    all_prompts = []
    all_results = []
    all_probs = []
    all_expected = []
    all_layers = []
    all_ranks = []
    
    # Iterate over each prompt
    for prompt in prompts_data:
        # If a prompt is an empty string, skip it
        prompt = prompt[0]
        if not prompt:
            continue
        
        # Run the lens model on the prompt
        lens_output = run_lens(llama, prompt)
        
        # Accumulate results for each layer
        for layer_idx in range(len(lens_output[1])):
            all_prompts.append(prompt)
            all_results.append(lens_output[0][layer_idx])
            all_probs.append(float(lens_output[1][layer_idx]))
            all_expected.append(lens_output[2])
            all_layers.append(int(layer_idx))
            all_ranks.append(int(lens_output[3][layer_idx]))

    # Create DataFrame from accumulated results
    prompts_with_probs = pd.DataFrame(
        {
            "prompt": all_prompts,
            "layer": all_layers,
            "results": all_results,
            "probs": all_probs,
            "expected": all_expected,
        })
    
    prompts_with_ranks = pd.DataFrame(
        {
            "prompt": all_prompts,
            "layer": all_layers,
            "results": all_results,
            "ranks": all_ranks,
            "expected": all_expected,
        })
    return plot_prob(prompts_with_probs), plot_rank(prompts_with_ranks)

def clear_all(prompts):
    prompts=[['']]
    # prompt_file=gr.File(type="filepath", label="Upload a File with Prompts")
    prompt_file = None
    prompts_data = gr.Dataframe(headers=["Prompt"], row_count=5, col_count=1, value= prompts, type="array", interactive=True)
    return prompts_data,prompt_file,plot_prob(prompts_with_probs),plot_rank(prompts_with_ranks)

def gradio_interface():
    with gr.Blocks(theme="gradio/monochrome") as demo:
        prompts = [['The Eiffel Tower is located in the city of'],['Vatican is located in the city of']]
        
        with gr.Row():
            with gr.Column(scale=3):
                model_dropdown = gr.Dropdown(choices=list(MODEL_OPTIONS.keys()), label="Select Model", value="Llama3.1-8B")
                prompts_data = gr.Dataframe(headers=["Prompt"], row_count=5, col_count=1, value= prompts, type="array", interactive=True)
            with gr.Column(scale=1):
                prompt_file=gr.File(type="filepath", label="Upload a File with Prompts")
        prompt_file.upload(process_file, inputs=[prompts_data,prompt_file], outputs=[prompts_data])
        # Define the outputs
        with gr.Row():
            clear_btn = gr.Button("Clear")
            submit_btn = gr.Button("Submit")
        
        prompt_file.upload(process_file, inputs=[prompts_data, prompt_file], outputs=[prompts_data])


        gr.Markdown("The most likely output token is the model's prediction at the final layer, shown in brackets in the plot legend.")
        # Create a Markdown component for the description
        with gr.Row():
            gr.Markdown("The graph below illustrates the probability of this most likely output token as it is decoded at each layer of the model. Each point on the graph is annotated with the decoded output corresponding to the token that has the highest probability at that particular layer.")
            gr.Markdown("The graph below illustrates the rank of this most likely output token as it is decoded at each layer of the model. Each point on the graph is annotated with the decoded output corresponding to the token that has the lowest rank at that particular layer.")

        prob_img, rank_img = submit_prompts(model_dropdown.value, prompts)

        with gr.Row():
            prob_visualization = gr.Image(value=prob_img, type="pil",label=" ")
            rank_visualization = gr.Image(value=rank_img, type="pil",label=" ")

        clear_btn.click(clear_all, inputs=[prompts_data], outputs=[prompts_data,prompt_file,prob_visualization,rank_visualization])
        submit_btn.click(submit_prompts, inputs=[model_dropdown,prompts_data], outputs=[prob_visualization,rank_visualization])#
        prompt_file.clear(clear_all, inputs=[prompts_data], outputs=[prompts_data,prompt_file,prob_visualization,rank_visualization])
        
        # Generate plots with sample prompts on load
        
        demo.launch()

gradio_interface()