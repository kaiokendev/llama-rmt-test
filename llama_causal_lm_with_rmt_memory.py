# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model and
# the implementation here:
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
from pathlib import Path
import traceback
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, List, Union
from torch.nn import CrossEntropyLoss
import transformers.models.llama.modeling_llama


class LlamaForCausalLMWithRMT(LlamaForCausalLM):
    def __init__(self, config, num_memory_vectors: int = 10, segment_length: int = 2048):
        super().__init__(config)
        self.num_memory_vectors = num_memory_vectors
        self.segment_length = segment_length
        self.memory_tokens = nn.Parameter(torch.randn(1, num_memory_vectors, config.hidden_size))

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        bsz, seq_len = input_ids.shape
        num_segments = (seq_len + self.segment_length - 1) // self.segment_length
        all_logits = []
        all_segment_outputs = []

        for segment_idx in range(num_segments):
            segment_start = segment_idx * self.segment_length
            segment_end = min((segment_idx + 1) * self.segment_length, seq_len)
            segment_input_ids = input_ids[:, segment_start:segment_end]
            segment_attention_mask = torch.ones(bsz, self.num_memory_vectors + seq_len)
            segment_position_ids = torch.arange(0, self.num_memory_vectors + seq_len, device=position_ids.device).unsqueeze(0)

            if segment_idx == 0:
                segment_memory = self.memory_tokens.expand(bsz, -1, -1)
                segment_input_ids = torch.cat([torch.full_like(segment_input_ids[:, :1], self.config.bos_token_id), segment_input_ids[:, :-1]], dim=1)
                segment_inputs_embeds = self.model.embed_tokens(segment_input_ids)
                segment_inputs_embeds = torch.cat([segment_memory, segment_inputs_embeds], dim=1)
            else:
                segment_inputs_embeds = self.model.embed_tokens(segment_input_ids)
                segment_inputs_embeds = torch.cat([updated_memory, segment_inputs_embeds], dim=1)

            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
            try:
                segment_outputs = self.model(
                    attention_mask=segment_attention_mask,
                    position_ids=segment_position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=segment_inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            except Exception:
                traceback.print_exc()

            segment_hidden_states = segment_outputs[0]
            segment_logits = self.lm_head(segment_hidden_states)

            if segment_idx < num_segments - 1:
                updated_memory = segment_logits[:, :self.num_memory_vectors, :]

            all_logits.append(segment_logits[:, self.num_memory_vectors:, :])
            all_segment_outputs.append(segment_outputs)

        logits = torch.cat(all_logits, dim=1)
        
        filtered_past_key_values = [segment_outputs.past_key_values for segment_outputs in all_segment_outputs if segment_outputs.past_key_values is not None]
        if len(filtered_past_key_values) > 1:
            all_past_key_values = torch.cat(filtered_past_key_values, dim=1)
        else:
            all_past_key_values = None
            
        filtered_hidden_states = [segment_outputs.hidden_states for segment_outputs in all_segment_outputs if segment_outputs.hidden_states is not None]
        if len(filtered_hidden_states) > 1:
            all_hidden_states = torch.cat(filtered_hidden_states, dim=1)
        else:
            all_hidden_states = None
            
        filtered_attentions = [segment_outputs.attentions for segment_outputs in all_segment_outputs if segment_outputs.attentions is not None]
        if len(filtered_attentions) > 1:
            all_attentions = torch.cat(filtered_attentions, dim=1)
        else:
            all_attentions = None
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits, all_past_key_values, all_hidden_states, all_attentions)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=all_past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
