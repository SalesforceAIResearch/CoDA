import os
import json
import warnings
import torch
import numpy as np
import torch.nn as nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset
from transformers import Trainer, ProcessorMixin
from torch import autograd, nn

from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import torch.distributions as dists
import tqdm
import torch.nn.functional as F
import wandb
from ...hparams import FinetuningArguments
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
import logging

def gaussian_kl(mu_p, sigma_p, mu_q, sigma_q):
    """KL(p||q)"""
    return (
        sigma_q.log() - sigma_p.log()
        + (sigma_p**2 + (mu_p - mu_q)**2)/(2*sigma_q**2)
        - 0.5
    )

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

class TorchPrimeDiffusionTrainer(Trainer):
    def __init__(
        self,
        finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ):
        super().__init__(**kwargs)
        # not used currently
        self.diff_args = finetuning_args
        self.finetuning_args = finetuning_args
        print("*"*30, self.diff_args)
        self.output_dir = self.args.output_dir
        self.noiser = LinearNoise()

        # Suppress deprecation warnings
        warnings.filterwarnings("ignore", message=".*Trainer.tokenizer is now deprecated.*")
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        logging.disable(logging.WARNING)

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: Optional[bool] = False,
        num_items_in_batch: Optional[int] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        r"""
        Computes diffusion loss.
        """
        # print(f"Current GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        final_loss = self.inner_forward(model, inputs)

        return final_loss

    def q_sample(self, x_0, t, maskable_mask):
        u = torch.rand_like(x_0, dtype=torch.float) # t/T prob to mask
        t_mask = (u < (t / self.diff_args.diffusion_steps)[:, None]) & maskable_mask
        x_t = x_0.masked_fill(t_mask, self.tokenizer.mask_token_id)
        return x_t, t, t_mask  #  True means it's "MASK" token and should have loss

    def q_sample_coupled(self, x_0, t1, t2, maskable_mask):
        # partial mask: True for the part should not be mask
        t1_eq_t2_mask = (t1 == t2)
        t1, t2 = torch.maximum(t1, t2).float(), torch.minimum(t1, t2).float()

        # sample t1
        u = torch.rand_like(x_0, dtype=torch.float)
        t1_mask = (u < (t1 / self.diff_args.diffusion_steps)[:, None]) & maskable_mask
        x_t1 = x_0.masked_fill(t1_mask, self.tokenizer.mask_token_id)

        # sample t2
        u = torch.rand_like(x_0, dtype=torch.float)
        t2_mask = t1_mask & (u > ((t1 - t2) / t1)[:, None])
        u = torch.rand_like(x_0[t1_eq_t2_mask], dtype=torch.float) 
        t2_mask[t1_eq_t2_mask] = (u < (t1[t1_eq_t2_mask] / self.diff_args.diffusion_steps)[:, None]) & (maskable_mask[t1_eq_t2_mask])
        x_t2 = x_0.masked_fill(t2_mask, self.tokenizer.mask_token_id)

        x_t  = torch.cat([x_t1, x_t2], dim=0)
        t = torch.cat([t1, t2])
        mask_mask = torch.cat([t1_mask, t2_mask], dim=0)
        return x_t, t, mask_mask  #  True means it's "MASK" token and should have loss

    def transition(self, x_0, sigma, maskable_mask):
        # move_chance = 1 - (-sigma).exp()
        move_chance = sigma
        move_indices = (torch.rand(*x_0.shape, device=x_0.device) < move_chance) & maskable_mask
        x_t = torch.where(move_indices, self.tokenizer.mask_token_id, x_0)
        return x_t

    def get_model_embeddings(self, model, x):
        """
        Get input embeddings from model, handling both DDP and regular models.
        This uses the get_embeds method that we implemented in the model.
        """
        if isinstance(model, DDP):
            return model.module.get_embeds(x)
        else:
            return model.get_embeds(x)

    def inner_forward(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        eval: bool = False,
    ):
        x = inputs["input_ids"]
        if "src_mask" not in inputs:
            raise ValueError("src_mask is required for SFT training")
        else:
            src_mask = inputs["src_mask"].bool()

        # Compute current epoch from training state
        global_step = int(getattr(self.state, "global_step", 0) or 0)
        total_steps = getattr(self.state, "max_steps", None)
        total_epochs = getattr(self.state, "num_train_epochs", None)
        if total_epochs is None:
            epoch = None         
        else:
            epoch = int(global_step / total_steps * total_epochs)

        if self.finetuning_args.progress_src_mask:
            # Progressively relax src_mask based on training progress.
            # At the beginning (global_step=0) -> all False; at progress_src_mask_ratio of total steps -> original src_mask.            
            if not total_steps or total_steps <= 0:
                total_steps = getattr(self.args, "max_steps", 0) or 0
            progress_src_mask_ratio = self.finetuning_args.progress_src_mask_ratio
            progress_src_mask_steps = max(1.0, float(total_steps) * progress_src_mask_ratio) if total_steps and total_steps > 0 else 1.0
            ratio = float(global_step) / progress_src_mask_steps
            ratio = float(max(0.0, min(1.0, ratio)))
            if global_step % 100 == 0:
                print(f"Global step: {global_step}, ratio: {ratio}")
            if ratio <= 0.0:
                # All False at the very beginning
                src_mask = torch.zeros_like(src_mask, dtype=torch.bool)
            elif ratio < 1.0:
                # Keep only the first floor(ratio * original_true_count) Trues per row,
                # turning the trailing Trues into False.
                batch_sz, seq_len_mask = src_mask.shape
                true_counts = src_mask.sum(dim=1)  # [batch]
                keep_len = torch.floor(true_counts.to(torch.float32) * ratio).to(torch.long)  # [batch]
                idx = torch.arange(seq_len_mask, device=src_mask.device).unsqueeze(0).expand(batch_sz, -1)
                src_mask = idx < keep_len.unsqueeze(1)
            # else ratio == 1.0 -> keep original src_mask

        batch_size, seq_len = x.shape
        # # Get vocab_size for the model
        # if isinstance(model, DDP):
        #     vocab_size = model.module.vocab_size
        # else:
        #     vocab_size = model.vocab_size
        # num_timesteps = self.diff_args.diffusion_steps

        #### countinous-time sampling
        # NOTE: Masking is now handled in the model's forward method to avoid double masking
        # sampling_eps = 1e-3
        # t = (1 - sampling_eps) * torch.rand(x.shape[0], device=x.device) + sampling_eps

        # sigma = self.noiser.total_noise(t)
        # dsigma = self.noiser.rate_noise(t)

        # x_t = self.transition(x, sigma[:, None], maskable_mask=~src_mask)
        #### full attention
        # attn_mask_ratio = min(1.0, (self.state.global_step + 1) / self.diff_args.anneal_steps)
        # attn_mask_ratio = 1.0

        # Use the fixed embedding access method
        # NOTE: Commenting out unnecessary embedding call - we only need dtype for attention mask
        # x_embed = self.get_model_embeddings(model, x)

        # NOTE (haolin): We don't need attention mask for TorchPrime models
        # attention_mask = get_anneal_attn_mask(seq_len, batch_size, dtype=x_embed.dtype, device=x.device, attn_mask_ratio=attn_mask_ratio)
        # Use torch.float32 for attention mask dtype (standard for attention computations)
        # attention_mask = get_anneal_attn_mask(seq_len, batch_size, dtype=torch.bfloat16, device=x.device, attn_mask_ratio=attn_mask_ratio)

        # attention_mask = torch.ones_like(x, dtype=torch.float)
        # Pass original input_ids and src_mask to model - masking will be handled in model's forward method
        logits, final_loss = model(x, attention_mask=None, src_mask=src_mask, training_mode="sft", epoch=epoch)

        # loss_mask = x_t == self.tokenizer.mask_token_id

        # if self.diff_args.shift:
        #     #### shift loss
        #     logits = logits[:,:-1]
        #     loss_mask = loss_mask[:,1:]
        #     x = x[:,1:]

        # loss = F.cross_entropy(logits.reshape(-1, vocab_size), x.reshape(-1), reduction="none").float().reshape(batch_size, -1)   # num_masked samples

        # loss = loss.masked_fill(~loss_mask, 0)
        # loss = (dsigma[:, None] * loss).sum() / (input_ids.shape[0] * input_ids.shape[1])
        # final_loss = loss #(dsigma[:, None] * loss).sum() / loss_mask.sum()   # avg token loss
        # unweighted_loss = (loss).sum() / loss_mask.sum()

        if ('LOCAL_RANK' not in os.environ) or (int(os.environ['LOCAL_RANK']) == 0):
            # print(eval, dsigma, final_loss.item(), src_mask.sum(dim=1), attn_mask_ratio)

            if not eval:
                wandb.log({
                    'total_loss': final_loss
                })
            else:
                wandb.log({
                    'eval_loss': final_loss
                })

        return final_loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        rewrite prediction_step for eval loss
        """
        model.eval()
        labels = inputs['input_ids']
        with torch.no_grad():
            # import pdb; pdb.set_trace();
            final_loss = self.inner_forward(model, inputs, eval=True)
            # generated_tokens = generate_samples(model, self.diff_args, self.tokenizer, inputs, True)
            generated_tokens = None

        # if ('LOCAL_RANK' not in os.environ) or (int(os.environ['LOCAL_RANK']) == 0):
        #     output_prediction_file = os.path.join(self.output_dir, f"generated_predictions_{self.state.global_step}.jsonl")
        #     with open(output_prediction_file, "a", encoding="utf-8") as writer:
        #         res: List[str] = []
        #         decoded_preds = self.tokenizer.batch_decode(generated_tokens.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
        #         decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        #         for pred, label in zip(decoded_preds, decoded_labels):
        #             res.append(json.dumps({"predict": pred, "label": label}, ensure_ascii=False))
        #         writer.write("\n".join(res))

        return final_loss, None, None

    def eval_forward(
        self,
        model: nn.Module,
        inputs: Dict[str, Union["torch.Tensor", Any]],
        eval: bool = False,
    ):
        x = inputs["input_ids"].to('cuda')
        if "src_mask" not in inputs:
            src_mask = torch.zeros_like(x, dtype=torch.bool).to('cuda')
        else:
            src_mask = inputs["src_mask"].bool().to('cuda')
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Get vocab_size for the model
        if isinstance(model, DDP):
            vocab_size = model.module.vocab_size
        else:
            vocab_size = model.vocab_size

        num_timesteps = self.diff_args.diffusion_steps

        total_loss = torch.tensor(0.0).to(x.device)
        total_unw_loss = torch.tensor(0.0).to(x.device)
        # import pdb;pdb.set_trace();

        for t in range(num_timesteps-1, 0, -1): # t from T-1 to 1
            with torch.no_grad():
                #### select rate% tokens to be still [MASK]
                # NOTE: Masking is now handled in the model's forward method to avoid double masking
                # rate = t / num_timesteps

                # tt = torch.Tensor([rate]*batch_size).to(x.device)

                # sigma = self.noiser.total_noise(tt)
                # dsigma = self.noiser.rate_noise(tt)

                # x_t = self.transition(x, sigma[:, None], maskable_mask=~src_mask)

                #### full attention
                attn_mask_ratio = 1.0
                # Use the fixed embedding access method
                # NOTE: Commenting out unnecessary embedding call - we only need dtype for attention mask
                # x_embed = self.get_model_embeddings(model, x)

                # attention_mask = get_anneal_attn_mask(seq_len, batch_size, dtype=x_embed.dtype, device=x.device, attn_mask_ratio=attn_mask_ratio)
                # Use torch.float32 for attention mask dtype (standard for attention computations)
                attention_mask = get_anneal_attn_mask(seq_len, batch_size, dtype=torch.bfloat16, device=x.device, attn_mask_ratio=attn_mask_ratio)

                # Pass original input_ids and src_mask to model - masking will be handled in model's forward method
                logits, final_loss = model(x, attention_mask=attention_mask, src_mask=src_mask)

                # loss_mask = x_t == self.tokenizer.mask_token_id

                # if self.diff_args.shift:
                #     #### shift loss
                #     logits = logits[:,:-1]
                #     loss_mask = loss_mask[:,1:]
                #     label = x[:,1:]

                # loss = F.cross_entropy(logits.reshape(-1, vocab_size), label.reshape(-1), reduction="none").float().reshape(batch_size, -1)   # num_masked samples

                # loss = loss.masked_fill(~loss_mask, 0)

                # final_loss = (dsigma[:, None] * loss).sum() / loss_mask.sum()   # avg token loss
                # unw_loss = loss.sum() / loss_mask.sum()

                total_loss += final_loss
                # total_unw_loss += unw_loss

        if ('LOCAL_RANK' not in os.environ) or (int(os.environ['LOCAL_RANK']) == 0):
            print('eval_loss', total_loss/num_timesteps)

            if not eval:
                wandb.log({
                    'total_loss': total_loss/num_timesteps
                })
            else:
                wandb.log({
                    'eval_loss': total_loss/num_timesteps,
                    # 'eval_unweighted_loss': total_unw_loss/num_timesteps
                })

        return total_loss/num_timesteps


def topk_masking(scores, cutoff_len, stochastic=False, temp=1.0):
    """
    scores: [b, n]
    cutoff_len: [b, 1]
    stochastic: bool, whether to add noise to select top_k or not
    returns:
        mask: [b, n], with 1 if the token is in top-k lowest scores, 0 otherwise
    """
    if stochastic:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
        _scores = scores + temp * gumbel_noise
    else:
        _scores = scores
    sorted_index = _scores.sort(-1)[0]
    cutoff = sorted_index.gather(dim=-1, index=cutoff_len) # + 1e-10
    # cutoff_len = k -> select k + 1 tokens
    masking = _scores < cutoff
    return masking

def transition(x_0, sigma, maskable_mask, mask_token_id):
    # move_chance = 1 - (-sigma).exp()
    move_chance = sigma
    move_indices = (torch.rand(*x_0.shape, device=x_0.device) < move_chance) & maskable_mask
    x_t = torch.where(move_indices, mask_token_id, x_0)
    return x_t

def q_sample(xt, x0, t, maskable_mask, mask_token_id):
    u = torch.rand_like(x0, dtype=torch.float) # t/T prob to mask
    t_mask = (u < t) & maskable_mask
    xt = x0.masked_fill(t_mask, mask_token_id)
    return xt

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

def generate_samples(model, diff_args, tokenizer, inputs, eval=False, gsm=False):
    """
        select 1/T% tokens to denoise at each step
    """
    # model.cuda()
    model.eval()
    print("*** Start sampling, top-k-mask...")

    logits_temp = diff_args.logits_temp
    if eval:
        x = inputs["input_ids"].to(model.device)
        if "src_mask" not in inputs:
            src_mask = torch.zeros_like(x, dtype=torch.bool).to(model.device)
        else:
            src_mask = inputs["src_mask"].bool().to(model.device)
    else:
        x = torch.transpose(torch.stack(inputs['input_ids']), 0, 1).cuda()
        # src_mask = torch.transpose(torch.stack(inputs['src_mask']), 0, 1).bool().cuda()
        if "src_mask" not in inputs:
            src_mask = torch.zeros_like(x, dtype=torch.bool).cuda()
        else:
            src_mask = torch.transpose(torch.stack(inputs['src_mask']), 0, 1).bool().cuda()

    # Use the get_embeds method
    if isinstance(model, DDP):
        x_embed = model.module.get_embeds(x)
    else:
        x_embed = model.get_embeds(x)
    
    seq_len = x.size(1)
    batch_size = x.size(0)
    attention_mask = get_anneal_attn_mask(seq_len, batch_size, dtype=x_embed.dtype, device=x.device, attn_mask_ratio=1.0)

    init_maskable_mask = maskable_mask = ~src_mask
    
    # first forward, all position except src is [M]
    xt = x.masked_fill(maskable_mask, tokenizer.mask_token_id)

    if not eval:
        print(f"t=T(in):", tokenizer.decode(xt.tolist()[0]))

    logits = model(xt, attention_mask=attention_mask)
    # filter_logits = top_p_logits(logits/logits_temp, p=0.9)
    filter_logits = logits/logits_temp
    scores = torch.log_softmax(filter_logits, dim=-1)
    if gsm:
        x0_scores, x0 = scores.max(-1)
        # print("max..")
    # scores = scores.to(torch.float16)
    else:
        x0 = dists.Categorical(logits=scores).sample()
        x0_scores = torch.gather(scores, -1, x0.unsqueeze(-1)).squeeze(-1)

    if diff_args.shift:
        #### deal with shift, left most token will be replaced anyway
        x0 = torch.cat([x[:,0:1], x0[:, :-1]], dim=1)
        x0_scores = torch.cat([x0_scores[:,0:1], x0_scores[:, :-1]], dim=1)
    
    #### replace output of non-[MASK] positions with xt
    x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])
    if not eval:
        print(f"t=T(out):", tokenizer.decode(x0.tolist()[0]))

    for t in range(diff_args.diffusion_steps-1, 0, -1): # t from T-1 to 1
        with torch.no_grad():
            #### select rate% tokens to be still [MASK]
            rate = t / diff_args.diffusion_steps

            # compute the cutoff length for denoising top-k positions
            cutoff_len = (init_maskable_mask.sum(1, keepdim=True) * rate).long()
            # set the scores of unmaskable symbols to a large value so that they will never be selected
            _scores_for_topk = x0_scores.masked_fill(~init_maskable_mask, 1000.0)
            
            # select cutoff_len (low confidence) tokens to be still [MASK], others is denoised
            lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=False)        

            masked_to_x0 = maskable_mask & ~lowest_k_mask
            
            xt.masked_scatter_(masked_to_x0, x0[masked_to_x0])
            maskable_mask = maskable_mask.masked_fill(~lowest_k_mask, False)
            
            if not eval:
                print(f"t={t}(in):", tokenizer.decode(xt.tolist()[0]))

            logits = model(xt, attention_mask=attention_mask)
            # filter_logits = top_p_logits(logits/logits_temp, p=0.9)
            filter_logits = logits/logits_temp
            scores = torch.log_softmax(filter_logits, dim=-1)
            # scores
            
            if gsm:
                x0_scores, x0 = scores.max(-1)
            # scores = scores.to(torch.float16)
            # import pdb; pdb.set_trace();
            else:
                x0 = dists.Categorical(logits=scores).sample()
                x0_scores = torch.gather(scores, -1, x0.unsqueeze(-1)).squeeze(-1)

            if diff_args.shift:
                #### deal with shift, left most token will be replaced anyway
                x0 = torch.cat([x[:,0:1], x0[:, :-1]], dim=1)
                x0_scores = torch.cat([x0_scores[:,0:1], x0_scores[:, :-1]], dim=1)
            
            # replace output of non-[MASK] positions with xt
            x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])
            if not eval:
                print(f"t={t}(out):", tokenizer.decode(x0.tolist()[0]))
    
    if diff_args.shift:
        x0 = x0[:,1:]

    return x0

def generate_samples_v2(model, diff_args, tokenizer, inputs, eval=False):
    """
        select 1/T% tokens to denoise at each step
    """
    # model.cuda()
    model.eval()
    print("*** Start sampling, random keep...")

    logits_temp = diff_args.logits_temp
    if eval:
        x = inputs["input_ids"].to(model.device)
        if "src_mask" not in inputs:
            src_mask = torch.zeros_like(x, dtype=torch.bool).to(model.device)
        else:
            src_mask = inputs["src_mask"].bool().to(model.device)
    else:
        x = torch.transpose(torch.stack(inputs['input_ids']), 0, 1).cuda()
        # src_mask = torch.transpose(torch.stack(inputs['src_mask']), 0, 1).bool().cuda()
        if "src_mask" not in inputs:
            src_mask = torch.zeros_like(x, dtype=torch.bool).cuda()
        else:
            src_mask = torch.transpose(torch.stack(inputs['src_mask']), 0, 1).bool().cuda()

    # Use the get_embeds method
    if isinstance(model, DDP):
        x_embed = model.module.get_embeds(x)
    else:
        x_embed = model.get_embeds(x)
    
    seq_len = x.size(1)
    batch_size = x.size(0)
    attention_mask = get_anneal_attn_mask(seq_len, batch_size, dtype=x_embed.dtype, device=x.device, attn_mask_ratio=1.0)

    # first forward, all position except src is [M]
    xt = x.masked_fill(maskable_mask, tokenizer.mask_token_id)

    if not eval:
        print(f"t=T(in):", tokenizer.decode(xt.tolist()[0]))

    logits = model(xt, attention_mask=attention_mask)
    filter_logits = top_p_logits(logits/logits_temp, p=0.9)
    # filter_logits = logits
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
    if not eval:
        print(f"t=T(out):", tokenizer.decode(x0.tolist()[0]))

    import time

    for t in range(diff_args.diffusion_steps-1, 0, -1): # t from T-1 to 1
        with torch.no_grad():
            start_time = time.time()
            #### select rate% tokens to be still [MASK]
            p_to_x0 = 1/(t+1)
            
            masked_to_x0 = maskable_mask & (torch.rand_like(x0, dtype=torch.float) < p_to_x0)
            xt.masked_scatter_(masked_to_x0, x0[masked_to_x0])
            maskable_mask = maskable_mask.masked_fill(masked_to_x0, False)
            
            if not eval:
                print(f"t={t}(in):", tokenizer.decode(xt.tolist()[0]))

            logits = model(xt, attention_mask=attention_mask)
            filter_logits = top_p_logits(logits/logits_temp, p=0.9)
            # filter_logits = logits
            scores = torch.log_softmax(filter_logits, dim=-1)

            x0 = dists.Categorical(logits=scores).sample()
            x0_scores = torch.gather(scores, -1, x0.unsqueeze(-1)).squeeze(-1)

            if diff_args.shift:
                #### deal with shift, left most token will be replaced anyway
                x0 = torch.cat([x[:,0:1], x0[:, :-1]], dim=1)
                x0_scores = torch.cat([x0_scores[:,0:1], x0_scores[:, :-1]], dim=1)
            
            # replace output of non-[MASK] positions with xt
            x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])
            if not eval:
                print(f"t={t}(out):", tokenizer.decode(x0.tolist()[0]))
                
            end_time = time.time()
            iteration_time = end_time - start_time
            # gpu_memory_allocated = torch.cuda.memory_allocated()
            # gpu_memory_reserved = torch.cuda.memory_reserved()

            # print(f"Iteration {t}: Time = {iteration_time:.4f} seconds, "
            #     f"GPU Memory Allocated = {gpu_memory_allocated / 1e6:.2f} MB, "
            #     f"GPU Memory Reserved = {gpu_memory_reserved / 1e6:.2f} MB")
            
    if diff_args.shift:
        x0 = x0[:,1:]

    return x0

def eval_forward(model, inputs, diff_args, tokenizer):
    model.eval()
    x = inputs["input_ids"].to(model.device)
    if "src_mask" not in inputs:
        src_mask = torch.zeros_like(x, dtype=torch.bool).to(model.device)
    else:
        src_mask = inputs["src_mask"].bool().to(model.device)
    batch_size = x.size(0)
    seq_len = x.size(1)
    noiser = LinearNoise()

    # Get vocab_size for the model
    if isinstance(model, DDP):
        vocab_size = model.module.vocab_size
    else:
        vocab_size = model.vocab_size

    num_timesteps = diff_args.diffusion_steps

    total_loss = torch.tensor(0.0).to(x.device)
    total_unw_loss = torch.tensor(0.0).to(x.device)

    for t in range(num_timesteps, 0, -1): # t from T-1 to 1
        with torch.no_grad():
            #### select rate% tokens to be still [MASK]
            rate = t / num_timesteps

            tt = torch.Tensor([rate]*batch_size).to(x.device)

            sigma = noiser.total_noise(tt)
            dsigma = noiser.rate_noise(tt)

            x_t = transition(x, sigma[:, None], ~src_mask, tokenizer.mask_token_id)

            #### full attention
            attn_mask_ratio = 1.0
            # Use the get_embeds method
            if isinstance(model, DDP):
                x_embed = model.module.get_embeds(x)
            else:
                x_embed = model.get_embeds(x)

            attention_mask = get_anneal_attn_mask(seq_len, batch_size, dtype=x_embed.dtype, device=x.device, attn_mask_ratio=attn_mask_ratio)

            logits = model(x_t, attention_mask=attention_mask)

            loss_mask = x_t == tokenizer.mask_token_id

            if diff_args.shift:
                #### shift loss
                logits = logits[:,:-1]
                loss_mask = loss_mask[:,1:]
                label = x[:,1:]

            loss = F.cross_entropy(logits.reshape(-1, vocab_size), label.reshape(-1), reduction="none").float().reshape(batch_size, -1)   # num_masked samples
            
            loss = loss.masked_fill(~loss_mask, 0)
            
            final_loss = (dsigma[:, None] * loss).sum() / loss_mask.sum()   # avg token loss
            unw_loss = loss.sum() / loss_mask.sum()
            
            if loss_mask.sum() == 0:
                final_loss = unw_loss = torch.tensor(0.0).to(x.device)

            total_loss += final_loss
            total_unw_loss += unw_loss

    return total_unw_loss/num_timesteps


def get_attn_mask(seq_len, bsz, dtype, device):
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask[None, None, :, :].expand(bsz, 1, seq_len, seq_len)

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
