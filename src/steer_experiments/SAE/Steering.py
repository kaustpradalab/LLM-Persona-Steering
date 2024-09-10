import os
import torch
from torch import Tensor
from transformer_lens import utils
from functools import partial
from jaxtyping import Int, Float
from transformer_lens import HookedTransformer
from sae_lens import SAE

def set_up():
    global device
    torch.set_grad_enabled(False)
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

def load_model(model_name, sae_name, layer):
    # get model
    model = HookedTransformer.from_pretrained(model_name, device = device)
    # get the SAE for this layer
    sae, cfg_dict, _ = SAE.from_pretrained(
        release = sae_name,
        sae_id = f"blocks.{layer}.hook_resid_post",
        device = device
    )
    # get hook point
    hook_point = sae.cfg.hook_name
    print(hook_point)
    return model, sae, hook_point

def create_steering_hook(coeff, steering_vector, steering_on):
    def steering_hook(resid_pre, hook):
        if resid_pre.shape[1] == 1:
            return
        position = sae_out.shape[1]
        if steering_on:
            resid_pre[:, :position - 1, :] += coeff * steering_vector
    return steering_hook

def hooked_generate(model, prompt_batch, fwd_hooks=[], seed=None, **kwargs):
    if seed is not None:
        torch.manual_seed(seed)

    with model.hooks(fwd_hooks=fwd_hooks):
        tokenized = model.to_tokens(prompt_batch)
        result = model.generate(
            stop_at_eos=False,  # avoids a bug on MPS
            input=tokenized,
            max_new_tokens=50,
            do_sample=True,
            **kwargs)
    return result

def apply_steering_vectors(resid_pre, steering_vectors, coeffs, position):
    for steering_vector, coeff in zip(steering_vectors, coeffs):
        resid_pre[:, :position - 1, :] += coeff * steering_vector

def run_generate(example_prompt, model, layer, coeff, steering_vector, steering_on, sampling_kwargs):
    model.reset_hooks()
    steering_hook = create_steering_hook(coeff, steering_vector, steering_on)
    editing_hooks = [(f"blocks.{layer}.hook_resid_post", steering_hook)]
    res = hooked_generate([example_prompt] * 3, editing_hooks, seed=None, **sampling_kwargs)
    # Print results, removing the ugly beginning of sequence token
    res_str = model.to_string(res[:, 1:])
    print(("\n\n" + "-" * 80 + "\n\n").join(res_str))


def main():
    set_up()
    model_name = "gemma-2b"
    sae_name = "gemma-2b-res-jb"
    layer = 6
    model, sae, hook_point = load_model(model_name, sae_name, layer)

    steering_vector = sae.W_dec[10200]
    question_prompt = "What is on your mind?"
    coeff = 100
    sampling_kwargs = dict(temperature=1.0, top_p=0.1, freq_penalty=1.0)
    steering_on = True
    run_generate(question_prompt, model, layer, coeff, steering_vector, steering_on, sampling_kwargs)

if __name__ == '__main__':
    main()