from util.steering import *

device = set_up()
#model_name = "gemma-2-2b-it"
#sae_name = "gemma-scope-2b-pt-res-canonical"
#sae_id = "layer_17/width_65k/canonical"
#model, sae = load_model(model_name, sae_name, sae_id, device)


model_name = "gemma-2-9b-it"
sae_name = "gemma-scope-9b-it-res-canonical"
sae_id = "layer_31/width_131k/canonical"
model, sae = load_model(model_name, sae_name, sae_id, device)