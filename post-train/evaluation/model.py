import torch
import torch.distributions as dists
from attention_patch import replace_attention_mask

replace_attention_mask()

import transformers

from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig
import torch.nn.functional as F
import torch.nn as nn
## add attention_patch
from huggingface_hub import PyTorchModelHubMixin

class DiscreteDiffusionModel(nn.Module, PyTorchModelHubMixin):
    """
    diffusion model
    """
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def __init__(
        self,
        model,
        config,
        tokenizer,
        device
    ):
        super().__init__()
        if isinstance(model, str): # if use pre-trained model name from huggingface, e.g., gpt2, gpt2-medium.
            config_pt = AutoConfig.from_pretrained(model)
            self.model = AutoModelForCausalLM.from_config(config_pt)
        else:
            self.model = model
        self.config = config
        self.embed_dim = self.config.hidden_size
        self.hidden_dim = self.config.hidden_size
        if self.model.get_input_embeddings().weight.size(0) != len(tokenizer):
            self.model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=2)
        self.vocab_size = self.model.get_input_embeddings().weight.size(0)
        if getattr(self.config, "model_type", None) == "gpt2":
            self.embed_tokens = self.model.transformer.wte
            self.denoise_model = self.model.transformer # use inputs_embeds instead of input_ids in forward function
            for gpt2block in self.model.transformer.h:
                gpt2block.attn.bias.fill_(True)  # remove causal mask
            self.lm_head = self.model.lm_head
            del self.denoise_model.wte
        elif getattr(self.config, "model_type", None) == "llama":
            self.embed_tokens = self.model.model.embed_tokens
            self.denoise_model = self.model.model
            self.lm_head = self.model.lm_head
            del self.denoise_model.embed_tokens
        del self.model
        self.device = device

    def get_logits(self, hidden_repr):
        return self.lm_head(hidden_repr)

    def get_input_embeddings(self):
        return self.embed_tokens
    
    def get_embeds(self, input_ids):
        return self.embed_tokens(input_ids)
    
    def forward(self, input_ids, attention_mask, inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        """
        denoise the input
        """
        x_embed = self.get_embeds(input_ids)

        x = self.denoise_model(inputs_embeds = x_embed, attention_mask=attention_mask, return_dict = False)[0]

        logits = self.get_logits(x)

        return logits

def generate_samples(model, diff_args, tokenizer, inputs, verbose=False):
    """
        select 1/T% tokens to denoise at each step
    """
    # model.cuda()
    model.eval()
    print("*** Start sampling, random keep...")

    logits_temp = diff_args.logits_temp
    topp_temp = diff_args.topp_temp

    x = inputs["input_ids"].to(model.device)
    if "src_mask" not in inputs:
        src_mask = torch.zeros_like(x, dtype=torch.bool).to(model.device)
    else:
        src_mask = inputs["src_mask"].bool().to(model.device)

    x_embed = model.get_embeds(x)
    seq_len = x.size(1)
    batch_size = x.size(0)
    attention_mask = get_anneal_attn_mask(seq_len, batch_size, dtype=x_embed.dtype, device=x.device, attn_mask_ratio=1.0) # all 0

    init_maskable_mask = maskable_mask = ~src_mask
    
    # first forward, all position except src is [M]
    xt = x.masked_fill(maskable_mask, tokenizer.mask_token_id)

    if verbose:
        print(f"t=T(in):", tokenizer.decode(xt.tolist()[0]))

    logits = model(xt, attention_mask=attention_mask)
    filter_logits = top_p_logits(logits/logits_temp, p=topp_temp)
    scores = torch.log_softmax(filter_logits, dim=-1)
    # x0_scores, x0 = scores.max(-1)
    # scores = scores.to(torch.float16)
    x0 = dists.Categorical(logits=scores).sample()
    x0_scores = torch.gather(scores, -1, x0.unsqueeze(-1)).squeeze(-1)

    if diff_args.shift:
        #### deal with shift, left most token will be replaced anyway
        x0 = torch.cat([x[:,0:1], x0[:, :-1]], dim=1)
        x0_scores = torch.cat([x0_scores[:,0:1], x0_scores[:, :-1]], dim=1)
    
    #### replace output of non-[MASK] positions with xt
    x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])
    if verbose:
        print(f"t=T(out):", tokenizer.decode(x0.tolist()[0]))

    for t in range(diff_args.diffusion_steps-1, 0, -1): # t from T-1 to 1
        with torch.no_grad():
            #### select rate% tokens to be still [MASK]
            p_to_x0 = 1/(t+1)
            
            masked_to_x0 = maskable_mask & (torch.rand_like(x0, dtype=torch.float) < p_to_x0)
            xt.masked_scatter_(masked_to_x0, x0[masked_to_x0])
            maskable_mask = maskable_mask.masked_fill(masked_to_x0, False)
            if verbose:
                print(f"t={t}(in):", tokenizer.decode(xt.tolist()[0]))

            logits = model(xt, attention_mask=attention_mask)
            filter_logits = top_p_logits(logits/logits_temp, p=topp_temp)
            scores = torch.log_softmax(filter_logits, dim=-1)
            x0 = dists.Categorical(logits=scores).sample()
            x0_scores = torch.gather(scores, -1, x0.unsqueeze(-1)).squeeze(-1)

            if diff_args.shift:
                #### deal with shift, left most token will be replaced anyway
                x0 = torch.cat([x[:,0:1], x0[:, :-1]], dim=1)
                x0_scores = torch.cat([x0_scores[:,0:1], x0_scores[:, :-1]], dim=1)
            
            # replace output of non-[MASK] positions with xt
            x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])
            if verbose:
                print(f"t={t}(out):", tokenizer.decode(x0.tolist()[0]))
            
    if diff_args.shift:
        x0 = x0[:,1:]

    return x0

class LinearNoise():
    """
    Linear noise schedule built so that alpha_t interpolates between 0 and ~1
    when t goes from 0 to 1. Used for absorbing

    """
    def __init__(self):
        super().__init__()

    def rate_noise(self, t): # weighting with (alpha_t)'/(1-alpha_t)
        return torch.reciprocal(t)

    def total_noise(self, t): # 0~1
        return t

def get_anneal_attn_mask(seq_len, bsz, dtype, device, attn_mask_ratio):
    mask = torch.full((seq_len, seq_len), 0, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 1)
    causal_mask = mask.to(dtype)
    
    random_mask = torch.bernoulli(torch.full((seq_len, seq_len), 0.0, device=device) + attn_mask_ratio)

    anneal_mask = torch.logical_or(causal_mask, random_mask)
    expanded_mask = anneal_mask[None, None, :, :].expand(bsz, 1, seq_len, seq_len)
    inverted_mask = 1.0 - expanded_mask.to(dtype)

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def top_p_logits(logits, p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    # import pdb; pdb.set_trace();
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits