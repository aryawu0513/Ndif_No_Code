from IPython.display import clear_output
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
from nnsight import LanguageModel

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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

# def unembed_tuned(model,results_list):
#     logit_list=[]
#     with torch.no_grad():
#         with model.trace('',remote=True) as runner:
#             for translated in results_list:
#                 normed_last_token_residual= model.model.norm(translated)
#                 logit= model.lm_head(normed_last_token_residual).save()
#                 # logit = model.model.unembed(translated)
#                 logit_list.append(logit)
#     results = torch.stack(logit_list)
#     # results.shape = (batch_size, n_layers, vocab_size)
#     results = results.transpose(1, 0)
#     return F.softmax(results, dim=2)

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

def decode_token(model,answer):
    top_tokens = torch.argmax(answer, dim=2)  # Shape: (batch_size, n_layers)

    # Decode the top tokens for each example in the batch and for each layer
    decoded_tokens_2d = np.array(
        [[model.tokenizer.decode(token.item()) for token in example_tokens] 
        for example_tokens in top_tokens]
    )
    return decoded_tokens_2d

def run_tuned_lens(model,lenses,PROMPT):
    residuals_by_layer,expected_token=run_prompt_get_residual(model,PROMPT)
    results_list=apply_tuned_transformation(residuals_by_layer,lenses)
    tuned_lens_results, tuned_lens_all_probs, actual_output,tuned_lens_ranks_by_layer=unembed_tuned(model,results_list,expected_token)
    return tuned_lens_results, tuned_lens_all_probs, actual_output,tuned_lens_ranks_by_layer

def main():
    MODEL_PATH="meta-llama/Meta-Llama-3.1-8B"
    llama = LanguageModel(MODEL_PATH)
    PROMPT = 'The Big Ben is located in the city of'
    residuals_by_layer,expected_token=run_prompt_get_residual(llama,PROMPT)
    actual_output = llama.tokenizer.decode(expected_token.item())
    print(actual_output)
    lens_checkpoint_path="/mnt/ssd/aryawu/prl_ml/finewebedu_10000_out"
    lenses = load_lenses(llama, lens_checkpoint_path)
    results_list=run_tuned_lens(residuals_by_layer,lenses)
    answer=unembed_tuned(llama,results_list)
    decoded_tokens_2d=decode_token(llama,answer)
    print(decoded_tokens_2d)

if __name__ == "__main__":
    main()