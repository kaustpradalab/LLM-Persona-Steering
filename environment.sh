# download the llama 3.1 model
pip install modelscope

modelscope download --model AI-ModelScope/Meta-Llama-3.1-8B

modelscope download --model LLM-Research/gemma-2-2b --local_dir '/root/autodl-tmp/gemma-2-2b'

# download the sae environments
pip install git+https://github.com/EleutherAI/sae.git

# update the huggingface mirror
pip install -U huggingface_hub

export HF_ENDPOINT=https://hf-mirror.com

# setting Environment var
export HF_DATASETS_CACHE=/root/autodl-tmp/cache
export HF_CACHE_DIR=/root/autodl-tmp/cache
export HF_HOME=/root/autodl-tmp/cache/huggingface
export HF_HUB_CACHE=/root/autodl-tmp/cache/huggingface/hub
export HF_ENDPOINT=https://hf-mirror.com 
export HF_TOKEN=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

# tokens
huggingface-cli login

llama3: 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
Gemma2: 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


#autodl tips
source /etc/network_turbo

import subprocess
import os

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value

unset http_proxy && unset https_proxy

#autodl cleaning
du -sh /tmp/
du -sh /root/.cache && rm -rf /root/.cache
du -sh /root/.local/share/Trash && rm -rf /root/.local/share/Trash 


source ~/.bashrc