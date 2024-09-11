import json

def apply_format(prompt, inference_type, tokenizer):
    if inference_type=="base":
        model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        model_inputs = model_inputs.input_ids
        return model_inputs
    elif inference_type=="chat":
        messages = [
            {"role": "user", "content": f"{prompt}"},]
        model_inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,return_tensors="pt").to("cuda")
        return model_inputs
    else:
        raise ValueError(f"inferente type should be either base or chat")
    


    

