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

#placeholder for reset
prompts_with_probs = pd.DataFrame(
{
    "prompt": ['waiting for data'],
    "layer": [0],
    "results": ['hi'],
    "probs": [0],
    "expected": ['hi'],
})
prompts_with_ranks = pd.DataFrame(
{
    "prompt": ['waiting for data'],
    "layer": [0],
    "ranks": [0],
    "expected": ['hi'],
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



#problem with using gr.LinePlot instead of a plt.figure is that text labels cannot be added for each individual point
def plot_prob(prompts_with_probs):
    return gr.LinePlot(prompts_with_probs, x="layer", y="probs",color="prompt", title="Probability of Expected Token",label="results",show_label=True,key="results")

def plot_rank(prompts_with_ranks):
    return gr.LinePlot(prompts_with_ranks, x="layer", y="ranks", color="prompt", title="Rank of Expected Token",label="expected",show_label=True,key="expected")

def plot_prob_mean(prompts_with_probs):
    summary_stats = prompts_with_probs.groupby("prompt")["probs"].agg(
        mean_prob="mean", 
        variance="var"
    ).reset_index()
    print("summary_stats",summary_stats)
    # Calculate the standard deviation for error bars
    summary_stats["std_dev"] = summary_stats["variance"] ** 0.5

    # Create the bar plot with error bars
    return gr.BarPlot(
        summary_stats,
        x="prompt",
        y="mean_prob",
        error_y="std_dev",
        title="Mean Probability of Expected Token",
    )

def plot_rank_mean(prompts_with_ranks):
    # Calculate mean and variance for each prompt
    summary_stats = prompts_with_ranks.groupby("prompt")["ranks"].agg(
        mean_rank="mean", 
        variance="var"
    ).reset_index()

    # Calculate the standard deviation for error bars
    summary_stats["std_dev"] = summary_stats["variance"] ** 0.5

    # Create the bar plot with error bars
    return gr.BarPlot(
        summary_stats,
        x="prompt",
        y="mean_rank",
        error_y="std_dev",
        title="Mean Rank of Expected Token",
    )

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
            "ranks": all_ranks,
            "expected": all_expected,
        })
    return plot_prob(prompts_with_probs), plot_rank(prompts_with_ranks),plot_prob_mean(prompts_with_probs),plot_rank_mean(prompts_with_ranks)


def clear_all(prompts):
    prompts=[['']]
    prompts_data = gr.Dataframe(headers=["Prompt"], row_count=5, col_count=1, value= prompts, type="array", interactive=True)
    return prompts_data,plot_prob(prompts_with_probs),plot_rank(prompts_with_ranks)


def gradio_interface():
    with gr.Blocks(theme="gradio/monochrome") as demo:
        prompts=[['']]
        prompts_data = gr.Dataframe(headers=["Prompt"], row_count=5, col_count=1, value= prompts, type="array", interactive=True)
        prompt_file=gr.File(type="filepath", label="Upload a File with Prompts")
        prompt_file.upload(process_file, inputs=[prompts_data,prompt_file], outputs=[prompts_data])

        # Define the outputs
        with gr.Row():
            prob_visualization = plot_prob(prompts_with_probs)
            rank_visualization = plot_rank(prompts_with_ranks)
        with gr.Row():
            prob_mean_visualization = plot_prob_mean(prompts_with_probs)
            rank_mean_visualization = plot_rank_mean(prompts_with_ranks)

        with gr.Row():
            clear_btn = gr.Button("Clear")
            clear_btn.click(clear_all, inputs=[prompts_data], outputs=[prompts_data,prob_visualization,rank_visualization])
            submit_btn = gr.Button("Submit")
            submit_btn.click(submit_prompts, inputs=[prompts_data], outputs=[prob_visualization,rank_visualization,prob_mean_visualization,rank_mean_visualization])
        

        demo.launch()


gradio_interface()