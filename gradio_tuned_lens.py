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
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Load the Language Model
# llama = LanguageModel("meta-llama/Meta-Llama-3.1-8B")

# Model options
MODEL_OPTIONS = {
    "Llama3.1-8B": "meta-llama/Meta-Llama-3.1-8B",
    # "Llama3.1-70B": "meta-llama/Meta-Llama-3.1-70B",
}


#placeholder for reset
prompts_with_probs = pd.DataFrame(
{
    "prompt": [''],
    "layer": [0],
    "tuned_results": [''],
    "tuned_probs": [0],
    "tuned_expected": [''],
    "logit_results": [''],
    "logit_probs": [0],
    "logit_expected": [''],
})
prompts_with_ranks = pd.DataFrame(
{
    "prompt": [''],
    "layer": [0],
    "tuned_results": [''],
    "tuned_ranks": [0],
    "tuned_expected": [''],
    "logit_results": [''],
    "logit_ranks": [0],
    "logit_expected": [''],
})


def run_prompt_get_residual(model,PROMPT):
    residuals_by_layer = []
    input_ids = model.tokenizer.encode(PROMPT)
    with model.trace(input_ids, remote=True) as runner:
        for layer_ix,layer in enumerate(model.model.layers):
            residual = layer.output[0][:, -1, :].save()
            residuals_by_layer.append(residual)
        tokens_out = model.lm_head.output.argmax(dim=-1).save()
        expected_token = tokens_out[0][-1].save()
    return residuals_by_layer,expected_token

import torch.nn as nn
class TunedLensTransformation(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.translator = nn.Linear(d_model, d_model)
        nn.init.eye_(self.translator.weight)
        self.translator.bias.data.zero_()

    def forward(self, x):
        self.translator.to(x.dtype)
        return self.translator(x)

def load_lenses(model, lens_checkpoint_path):
    num_layers = len(model.model.layers)
    lenses = []
    for i in range(num_layers):
        lens = TunedLensTransformation(model.config.hidden_size)
        lens.load_state_dict(
            torch.load(f"{lens_checkpoint_path}/tuned_lens_trained_{i}.pth")
        )
        lens.eval()
        lenses.append(lens)

    return lenses

def apply_tuned_transformation(residuals_by_layer,lenses):
    results_list = []
    for layer_idx,residual in enumerate(residuals_by_layer):
        translated = lenses[layer_idx](residual)
        results_list.append(translated)
    return results_list

def unembed_tuned(model,results_list,expected_token):
    tuned_lens_token_result_by_layer = []
    tuned_lens_probs_by_layer = []
    tuned_lens_ranks_by_layer = []
    with torch.no_grad():
        with model.trace('',remote=True) as tracer:
            for translated in results_list:
                normed_last_token_residual= model.model.norm(translated)
                tuned_lens_token_distribution= model.lm_head(normed_last_token_residual).save()
                tuned_lens_last_token_logit = tuned_lens_token_distribution[-1:]
                tuned_lens_probs = F.softmax(tuned_lens_last_token_logit, dim=1).save()
                tuned_lens_probs_by_layer.append(tuned_lens_probs)
                tuned_lens_next_token = torch.argmax(tuned_lens_probs, dim=1).save()
                tuned_lens_token_result_by_layer.append(tuned_lens_next_token)
    tuned_lens_all_probs = np.concatenate([probs[:, expected_token].cpu().detach().to(torch.float32).numpy() for probs in tuned_lens_probs_by_layer])

    for layer_probs in tuned_lens_probs_by_layer:
        # Sort the probabilities in descending order and find the rank of the expected token
        sorted_probs, sorted_indices = torch.sort(layer_probs, descending=True)
        # Find the rank of the expected token (1-based rank)
        expected_token_rank = (sorted_indices == expected_token).nonzero(as_tuple=True)[1].item() + 1
        tuned_lens_ranks_by_layer.append(expected_token_rank)
    actual_output = model.tokenizer.decode(expected_token.item())
    tuned_lens_results = [model.tokenizer.decode(next_token.item()) for next_token in tuned_lens_token_result_by_layer]
    return tuned_lens_results, tuned_lens_all_probs, actual_output,tuned_lens_ranks_by_layer

def run_tuned_lens(model,PROMPT,lenses):
    residuals_by_layer,expected_token=run_prompt_get_residual(model,PROMPT)
    results_list=apply_tuned_transformation(residuals_by_layer,lenses)
    tuned_lens_results, tuned_lens_all_probs, actual_output,tuned_lens_ranks_by_layer=unembed_tuned(model,results_list,expected_token)
    return tuned_lens_results, tuned_lens_all_probs, actual_output,tuned_lens_ranks_by_layer

def run_logit_lens(model,PROMPT):
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

import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
from adjustText import adjust_text

def plot_prob(prompts_with_probs):
    plt.figure(figsize=(10, 6))
    texts = []

    # Get unique prompts and prepare colors
    unique_prompts = prompts_with_probs['prompt'].unique()
    num_prompts = len(unique_prompts)

    # Define a colormap for the odd-indexed prompts
    colormap = plt.cm.get_cmap('tab10', num_prompts)  # Half the number of colors for odd indexed prompts

    # Iterate over each unique prompt and plot its probabilities
    for i, prompt in enumerate(unique_prompts):
        # Filter the DataFrame for the current prompt
        prompt_data = prompts_with_probs[prompts_with_probs['prompt'] == prompt]
        main_color = colormap(i)  # Color for odd-indexed prompts
        contrast_color = (main_color[0], main_color[1], main_color[2], 0.3)  # Lighten the color by changing alpha

        plt.plot(prompt_data['layer'], prompt_data['tuned_probs'], marker='x', color=main_color, label=f"{prompt} (Tuned)")
        plt.plot(prompt_data['layer'], prompt_data['logit_probs'], marker='o', color=contrast_color, linestyle='--', label=f"{prompt} (Logit)")

        # Annotate each point with the corresponding result
        for layer, prob, result in zip(prompt_data['layer'], prompt_data['tuned_probs'], prompt_data['tuned_results']):
            text = plt.text(layer, prob, result, fontsize=8)
            texts.append(text)
        
        for layer, prob, result in zip(prompt_data['layer'], prompt_data['logit_probs'], prompt_data['logit_results']):
            text = plt.text(layer, prob, result, fontsize=8)
            texts.append(text)

    # Add labels and title
    plt.xlabel('Layer Number')
    plt.ylabel('Probability')
    plt.title('Probability of Most-Likely Output Token')
    plt.grid(True)
    plt.xlim(0, max(prompts_with_probs['layer']))
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
    texts = []  # List to hold text annotations for adjustment

    unique_prompts = prompts_with_ranks['prompt'].unique()
    num_prompts = len(unique_prompts)

    colormap = plt.cm.get_cmap('tab10', num_prompts)

    # Iterate over each unique prompt and plot its ranks
    for i, prompt in enumerate(unique_prompts):
        # Filter the DataFrame for the current prompt
        prompt_data = prompts_with_ranks[prompts_with_ranks['prompt'] == prompt]

        main_color = colormap(i)
        contrast_color = (main_color[0], main_color[1], main_color[2], 0.3)

        plt.plot(prompt_data['layer'], prompt_data['tuned_ranks'], marker='x', color=main_color, label=f"{prompt} (Tuned)")
        plt.plot(prompt_data['layer'], prompt_data['logit_ranks'], marker='o', color=contrast_color, linestyle='--', label=f"{prompt} (Logit)")

        # Annotate each point with the corresponding result
        for layer, rank, result in zip(prompt_data['layer'], prompt_data['tuned_ranks'], prompt_data['tuned_results']):
            text = plt.text(layer, rank, result, ha='right', va='bottom', fontsize=8)
            texts.append(text)
        
        for layer, rank, result in zip(prompt_data['layer'], prompt_data['logit_ranks'], prompt_data['logit_results']):
            text = plt.text(layer, rank, result, ha='right', va='bottom', fontsize=8)
            texts.append(text)

    # Add labels and title
    plt.xlabel('Layer Number')
    plt.ylabel('Rank')
    plt.title('Rank of Most-Likely Output Token')
    plt.grid(True)
    plt.xlim(0, max(prompts_with_ranks['layer']))
    plt.ylim(bottom=0)
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


def submit_prompts(model_name, prompts_data):
    llama = LanguageModel(MODEL_OPTIONS[model_name])
    lens_checkpoint_path="/mnt/ssd/aryawu/prl_ml/finewebedu_10000_out"
    lenses = load_lenses(llama, lens_checkpoint_path)
    # Initialize lists to accumulate results
    all_prompts = []
    tuned_all_results = []
    tuned_all_probs = []
    tuned_all_expected = []
    all_layers = []
    tuned_all_ranks = []
    logit_all_results = []
    logit_all_probs = []
    logit_all_expected = []
    logit_all_ranks = []
    
    # Iterate over each prompt
    for prompt in prompts_data:
        # If a prompt is an empty string, skip it
        prompt = prompt[0]
        if not prompt:
            continue
        
        # Run the lens model on the prompt
        tuned_lens_output = run_tuned_lens(llama, prompt, lenses)
        logit_lens_output = run_logit_lens(llama, prompt)
        
        # Accumulate results for each layer
        for layer_idx in range(len(tuned_lens_output[1])):
            all_prompts.append(prompt)
            tuned_all_results.append(tuned_lens_output[0][layer_idx])
            logit_all_results.append(logit_lens_output[0][layer_idx])
            tuned_all_probs.append(float(tuned_lens_output[1][layer_idx]))
            logit_all_probs.append(float(logit_lens_output[1][layer_idx]))
            tuned_all_expected.append(tuned_lens_output[2])
            logit_all_expected.append(logit_lens_output[2])
            all_layers.append(int(layer_idx))
            tuned_all_ranks.append(int(tuned_lens_output[3][layer_idx]))
            logit_all_ranks.append(int(logit_lens_output[3][layer_idx]))
    # Create DataFrame from accumulated results
    prompts_with_probs = pd.DataFrame(
        {
            "prompt": all_prompts,
            "layer": all_layers,
            "tuned_results": tuned_all_results,
            "tuned_probs": tuned_all_probs,
            "tuned_expected": tuned_all_expected,
            "logit_results": logit_all_results,
            "logit_probs": logit_all_probs,
            "logit_expected": logit_all_expected,
        })
    
    prompts_with_ranks = pd.DataFrame(
        {
            "prompt": all_prompts,
            "layer": all_layers,
            "tuned_results": tuned_all_results,
            "tuned_ranks": tuned_all_ranks,
            "tuned_expected": tuned_all_expected,
            "logit_results": logit_all_results,
            "logit_ranks": logit_all_ranks,
            "logit_expected": logit_all_expected,
        })
    return plot_prob(prompts_with_probs), plot_rank(prompts_with_ranks)

def clear_all(prompts):
    prompts=[['']]
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


        gr.Markdown("This Demo shows a tuned lens trained on fineweb-Edu comparing with a logit lens.\n The model's prediction at the final layer is the most likely output token, shown in brackets in the plot legend.")
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