# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
import os
from typing import Dict, Tuple
from functools import partial
import torch
import torch.nn.functional as F
from megatron.training import get_args, get_model
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.training.checkpointing import load_checkpoint
from megatron.training.utils import average_losses_across_data_parallel_group
from megatron.training.global_vars import set_args
from mindspeed_llm.tasks.posttrain.base import BaseTrainer
from mindspeed_llm.tasks.posttrain.sft.sft_trainer import SFTTrainer
from mindspeed_llm.tasks.posttrain.utils import compute_log_probs
from megatron.training import get_args, get_tokenizer
from megatron.core import mpu, tensor_parallel
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    average_losses_across_data_parallel_group
)
from megatron.training import get_timers
from mindspeed_llm.training.utils import get_tune_attention_mask, get_finetune_data_on_this_tp_rank
from mindspeed_llm.tasks.posttrain.base import BaseTrainer
from mindspeed_llm.training.utils import  set_mtp_batch_list
from mindspeed_llm.core.transformer.multi_token_prediction import generate_mtp_batch_list_on_this_tp_rank
from mindspeed.core.context_parallel.get_batch_utils import set_actual_seq_len, get_ring_degree
from mindspeed.core.context_parallel.utils import pad_data

IGNORE_INDEX = -100

class OPGTrainer(BaseTrainer):
    """
    A trainer class for Offline Policy Gradient.

    This class provides methods for model initialize, computing losses and metrics, and training.
    """    

    def __init__(self):        
        super().__init__()        
    
    @staticmethod
    def get_batch(data_iterator):
        """Generate a batch."""
        # Items and their type.
        keys = ['input_ids', 'attention_mask', 'labels', 'reward']
        args = get_args()
        if args.reset_position_ids:
            keys += ['position_ids']
        data_type = torch.int64

        if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
            if args.variable_seq_lengths and args.pipeline_model_parallel_size > 2:
                tokens, attention_mask = get_finetune_data_on_this_tp_rank(data_iterator)
                return tokens, None, None, attention_mask, None, None
            else:
                # Broadcast data.
                data_b = tensor_parallel.broadcast_data(keys, next(data_iterator), data_type)
                if args.reset_position_ids:
                    generate_actual_seq_len(data_b)
                    batch = {'attention_mask': None}
                else:
                    attention_mask_1d = data_b.get('attention_mask').long()
                    attention_mask = get_tune_attention_mask(attention_mask_1d)
                    batch = {'attention_mask': attention_mask}
                batch = get_batch_on_this_cp_rank(batch)
                return None, None, None, batch['attention_mask'], None, None

        # Broadcast data.
        data_b = tensor_parallel.broadcast_data(keys, next(data_iterator), data_type)

        # Unpack
        labels = data_b.get('labels').long()
        tokens = data_b.get('input_ids').long()
        attention_mask_1d = data_b.get('attention_mask').long()
        reward = data_b.get('reward').long()
        # ignored label -100
        loss_mask = torch.where(labels == IGNORE_INDEX, 0, 1)
    
        if get_args().spec is not None and args.spec[0] == "mindspeed_llm.tasks.models.spec.hunyuan_spec":
            input_ids = tokens
            pad_id = 127961

            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

            loss_mask = torch.where(labels == IGNORE_INDEX, 0, 1)
            attention_mask = input_ids.ne(pad_id)

            position_ids = None
            batch = {
                'tokens': input_ids,
                'labels': labels,
                'loss_mask': loss_mask,
                'attention_mask': attention_mask,
                'position_ids': position_ids
            }
        else:
            if args.reset_position_ids:
                position_ids = data_b.get('position_ids').long()
                generate_actual_seq_len(data_b)

                batch = {
                    'tokens': tokens,
                    'labels': labels,
                    'loss_mask': loss_mask,
                }
                batch = get_batch_on_this_cp_rank(batch)
                batch['attention_mask'] = None
                batch['position_ids'] = position_ids
                return batch.values()

            attention_mask = get_tune_attention_mask(attention_mask_1d)
            position_ids = None
            batch = {
                    'tokens': tokens,
                    'labels': labels,
                    'loss_mask': loss_mask,
                    'attention_mask': attention_mask,
                    'position_ids': position_ids,
                    "reward": reward
                }
                # get batch_list for mtp_block
        if args.mtp_num_layers:
            mtp_batch_list = generate_mtp_batch_list_on_this_tp_rank(batch)
            set_mtp_batch_list(mtp_batch_list)
        batch = get_batch_on_this_cp_rank(batch)
        return batch.values()

    def forward_step(self, data_iterator, model):
        """PG Forward training step.

        Args:
            data_iterator : Input data iterator
            model (GPTModel): The GPT Model
        """
        # Get the batch.
        self.timers('batch-generator', log_level=2).start()        
        tokens, labels, loss_mask, attention_mask, position_ids, reward = self.get_batch(
            data_iterator)
        self.timers('batch-generator').stop()
                  
        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=labels, loss_mask=loss_mask)

        return output_tensor, partial(self.loss_func, loss_mask, reward)
    
    def loss_func(self, input_tensor: torch.Tensor, reward: torch.float, output_tensor: torch.Tensor):
        """Offline Policy Gradient Loss function.

        Args:
            input_tensor (torch.Tensor): The tensor with the labels (repeated in double)
            output_tensor (torch.Tensor): The tensor with the Policy Model's Logits
        """

        args = get_args()

        loss_mask = input_tensor
        loss_mask = loss_mask[..., 1:].view(-1).float()
        losses = output_tensor.float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        prob = torch.exp(-loss)
        loss = reward[0]*(1-prob) + (1-reward[0])*prob
                
        # Check individual rank losses are not NaN prior to DP all-reduce.
        if args.check_for_nan_in_loss_and_grad:
            global_rank = torch.distributed.get_rank()
            if loss.isnan():
                raise ValueError(f'Rank {global_rank}: found NaN in local forward loss calculation. '
                                 f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}')

        # Reduce loss for logging.
        averaged_loss = average_losses_across_data_parallel_group([loss])

        metrics = {}        
        metrics['reward'] = prob.detach().mean() * reward[0] + 1e-10
        metrics['lm loss'] = averaged_loss[0]

        return loss, metrics