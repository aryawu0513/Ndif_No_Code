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
llama = LanguageModel("meta-llama/Meta-Llama-3.1-8B")

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
    # logits_lens_all_probs = np.concatenate([probs[:, expected_token].cpu().detach().numpy() for probs in logits_lens_probs_by_layer])
    logits_lens_all_probs = np.concatenate([probs[:, expected_token].cpu().detach().to(torch.float32).numpy() for probs in logits_lens_probs_by_layer])

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
    
    # Iterate over each prompt and plot its probabilities
    for prompt in prompts_with_probs['prompt'].unique():
        # Filter the DataFrame for the current prompt
        prompt_data = prompts_with_probs[prompts_with_probs['prompt'] == prompt]
        
        # Plot probabilities for this prompt
        plt.plot(prompt_data['layer'], prompt_data['probs'], marker='x', label=prompt)
        
        # Annotate each point with the corresponding result
        for layer, prob, result in zip(prompt_data['layer'], prompt_data['probs'], prompt_data['results']):
            plt.text(layer, prob, result, fontsize=8)


    # Add labels and title
    plt.xlabel('Layer Number')
    plt.ylabel('Probability of Expected Token')
    plt.title('Prob of expected token across layers\n(annotated with actual decoded output at each layer)')
    plt.grid(True)
    plt.ylim(0.0, 1.0)
    plt.legend(title='Prompts', bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=1)

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')  # Use bbox_inches to avoid cutting off labels
    buf.seek(0)
    img = Image.open(buf)
    plt.close()  # Close the figure to free memory
    return img

def plot_rank(prompts_with_ranks):
    plt.figure(figsize=(10, 6))
    
    # Iterate over each prompt and plot its ranks
    for prompt in prompts_with_ranks['prompt'].unique():
        # Filter the DataFrame for the current prompt
        prompt_data = prompts_with_ranks[prompts_with_ranks['prompt'] == prompt]
        
        # Plot ranks for this prompt
        plt.plot(prompt_data['layer'], prompt_data['ranks'], marker='x', label=prompt)
        
        # Annotate each point with the corresponding result
        for layer, rank, result in zip(prompt_data['layer'], prompt_data['ranks'], prompt_data['results']):
            plt.text(layer, rank,result, ha='right', va='bottom', fontsize=8)

    # Add labels and title
    plt.xlabel('Layer Number')
    plt.ylabel('Rank of Expected Token')
    plt.title('Rank of expected token across layers\n(annotated with decoded output at each layer)')
    plt.grid(True)
    plt.ylim(bottom=0)  # Adjust if needed, depending on your rank values
    plt.legend(title='Prompts', bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=1)


    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')  # Use bbox_inches to avoid cutting off labels
    buf.seek(0)
    img = Image.open(buf)
    plt.close()  # Close the figure to free memory
    return img

def plot_prob_mean(prompts_with_probs):
    # Calculate mean probabilities and variance
    summary_stats = prompts_with_probs.groupby("prompt")["probs"].agg(
        mean_prob="mean", 
        variance="var"
    ).reset_index()

    # Set up the bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(summary_stats['prompt'], summary_stats['mean_prob'], 
                   yerr=summary_stats['variance']**0.5,  # Error bars are the standard deviation
                   capsize=5, color='skyblue')

    # Add labels and title
    plt.xlabel('Prompt')
    plt.ylabel('Mean Probability')
    plt.title('Mean Probability of Expected Token')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    plt.ylim(0, 1)


    # Annotate the mean and variance on the bars
    for bar, mean, var in zip(bars, summary_stats['mean_prob'], summary_stats['variance']):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f'Mean: {mean:.2f}\nVar: {var:.2f}', 
                 ha='center', va='bottom', fontsize=8, color='black')

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')  # Use bbox_inches to avoid cutting off labels
    buf.seek(0)
    img = Image.open(buf)
    plt.close()  # Close the figure to free memory
    return img

def plot_rank_mean(prompts_with_ranks):
    # Calculate mean ranks and variance
    summary_stats = prompts_with_ranks.groupby("prompt")["ranks"].agg(
        mean_rank="mean", 
        variance="var"
    ).reset_index()

    # Set up the bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(summary_stats['prompt'], summary_stats['mean_rank'], 
                   yerr=summary_stats['variance']**0.5,  # Error bars are the standard deviation
                   capsize=5, color='salmon')

    # Add labels and title
    plt.xlabel('Prompt')
    plt.ylabel('Mean Rank')
    plt.title('Mean Rank of Expected Token')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')

    # Annotate the mean and variance on the bars
    for bar, mean, var in zip(bars, summary_stats['mean_rank'], summary_stats['variance']):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f'Mean: {mean:.2f}\nVar: {var:.2f}', 
                 ha='center', va='bottom', fontsize=8, color='black')

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')  # Use bbox_inches to avoid cutting off labels
    buf.seek(0)
    img = Image.open(buf)
    plt.close()  # Close the figure to free memory
    return img

def submit_prompts(prompts_data):
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
    return plot_prob(prompts_with_probs), plot_rank(prompts_with_ranks),plot_prob_mean(prompts_with_probs),plot_rank_mean(prompts_with_ranks)


def clear_all(prompts):
    prompts=[['']]
    # prompt_file=gr.File(type="filepath", label="Upload a File with Prompts")
    prompt_file = None
    prompts_data = gr.Dataframe(headers=["Prompt"], row_count=5, col_count=1, value= prompts, type="array", interactive=True)
    return prompts_data,prompt_file,plot_prob(prompts_with_probs),plot_rank(prompts_with_ranks),plot_prob_mean(prompts_with_probs),plot_rank_mean(prompts_with_ranks)


def gradio_interface():
    with gr.Blocks(theme="gradio/monochrome") as demo:
        prompts=[['']]
        with gr.Row():
            with gr.Column(scale=3):
                prompts_data = gr.Dataframe(headers=["Prompt"], row_count=5, col_count=1, value= prompts, type="array", interactive=True)
            with gr.Column(scale=1):
                prompt_file=gr.File(type="filepath", label="Upload a File with Prompts")
        prompt_file.upload(process_file, inputs=[prompts_data,prompt_file], outputs=[prompts_data])
        # Define the outputs
        with gr.Row():
            clear_btn = gr.Button("Clear")
            submit_btn = gr.Button("Submit")
        with gr.Row():
            prob_visualization = gr.Image(value=plot_prob(prompts_with_probs), type="pil",label=" ")
            rank_visualization = gr.Image(value=plot_rank(prompts_with_ranks), type="pil",label=" ")
        with gr.Row():
            prob_mean_visualization = gr.Image(value=plot_prob_mean(prompts_with_probs), type="pil",label=" ")
            rank_mean_visualization = gr.Image(value=plot_rank_mean(prompts_with_ranks), type="pil",label=" ")

        clear_btn.click(clear_all, inputs=[prompts_data], outputs=[prompts_data,prompt_file,prob_visualization,rank_visualization,prob_mean_visualization,rank_mean_visualization])
        submit_btn.click(submit_prompts, inputs=[prompts_data], outputs=[prob_visualization,rank_visualization,prob_mean_visualization,rank_mean_visualization])#
        prompt_file.clear(clear_all, inputs=[prompts_data], outputs=[prompts_data,prompt_file,prob_visualization,rank_visualization,prob_mean_visualization,rank_mean_visualization])
        

        demo.launch()

gradio_interface()