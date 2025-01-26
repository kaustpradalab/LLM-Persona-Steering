import os
import torch
from torch import Tensor
from transformer_lens import utils
from functools import partial
from jaxtyping import Int, Float
from transformer_lens import HookedTransformer
from sae_lens import SAE

def set_up():
    torch.set_grad_enabled(False)
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    return device

def load_model(model_name, sae_name, sae_id, device):
    # get model
    model = HookedTransformer.from_pretrained(model_name, device = device)
    # get the SAE for this layer
    sae, cfg_dict, _ = SAE.from_pretrained(
        release = sae_name,
        sae_id = sae_id,
        device = device
    )
    return model, sae

def create_steering_hook(coeff, steering_vectors, steering_on):
    torch.set_grad_enabled(False)
    def steering_hook(resid_pre, hook):
        if resid_pre.shape[1] == 1:
            return
        position = resid_pre.shape[1]
        if steering_on:
            for steering_vector in steering_vectors:
                resid_pre[:, :position - 1, :] += coeff * steering_vector
    return steering_hook

def hooked_logit(model, input_ids, fwd_hooks=[], seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    with model.hooks(fwd_hooks=fwd_hooks):
        with torch.no_grad():
            outputs = model(input_ids)
        logits = outputs[:, -1, :]  # Logits for the last token
        probabilities = torch.softmax(logits, dim=-1)
    return probabilities

def get_likelihood_steer(input_ids, model, layer, coeff, steering_vectors, steering_on, seed=None):
    model.reset_hooks()
    steering_hook = create_steering_hook(coeff, steering_vectors, steering_on)
    editing_hooks = [(f"blocks.{layer}.hook_resid_post", steering_hook)]
    return hooked_logit(model, input_ids, editing_hooks, seed=seed)

def hooked_generate(model, inputs, fwd_hooks=[], seed=None, **kwargs):
    if seed is not None:
        torch.manual_seed(seed)
    with model.hooks(fwd_hooks=fwd_hooks):
        outputs = model.generate(input=inputs['input_ids'], do_sample=False, max_new_tokens=64, stop_at_eos=False, **kwargs)
    return outputs

def get_likelihood_generate(inputs, model, layer, coeff, steering_vectors, steering_on, sampling_kwargs, seed=None):
    model.reset_hooks()
    steering_hook = create_steering_hook(coeff, steering_vectors, steering_on)
    editing_hooks = [(f"blocks.{layer}.hook_resid_post", steering_hook)]
    return hooked_generate(model, inputs, editing_hooks, seed=seed, **sampling_kwargs)
    
def get_steer_vectors(sae, bg_type, features):
    if bg_type == "fixed":
        idx_dict = {}
        vector_list = []
        for category, indices in features.get("fixed", {}).items():
            idx_dict[category] = indices
            for idx in indices:
                vector_list.append(sae.W_dec[idx])
        return idx_dict, vector_list
        #steering_vectors = [sae.W_dec[5361], sae.W_dec[4365]]
    elif bg_type == "gen":
        pass