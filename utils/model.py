import torch
from torch import nn
import torch.nn.functional as F
from transformers import (
    ElectraConfig, 
    ElectraForMaskedLM, 
    PreTrainedModel,
    ElectraForPreTraining, 
    ElectraPreTrainedModel,
    Trainer
)
from transformers.models.electra.modeling_electra import ElectraForPreTrainingOutput

class ElectraForPretrainingModel(PreTrainedModel):
    def __init__(self, config_generator, config_discriminator, loss_weights=(1,50)):
        super().__init__(config_discriminator)

        self.generator = ElectraForMaskedLM(config_generator)
        self.discriminator = ElectraForPreTraining(config_discriminator)
        # weight sharing
        self.discriminator.electra.embeddings = self.generator.electra.embeddings
        self.generator.generator_lm_head.weight = self.generator.electra.embeddings.word_embeddings.weight
        self.init_weights()
        self.loss_weights = loss_weights

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
            self, 
            input_ids, labels, attention_mask=None, token_type_ids=None, 
            position_ids=None, head_mask=None, inputs_embeds=None, 
            output_attentions=None, 
            output_hidden_states=None, return_dict=None
        ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs_gen = self.generator(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
            position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, 
            labels=labels, output_attentions=False, 
            output_hidden_states=False, return_dict=True
        )

        loss_gen = outputs_gen.loss # (1,)
        logits_gen = outputs_gen.logits # (batch_size, seq_length, config.vocab_size)
        masked_bool = (labels == -100)
        # logits_masked: (batch_size*masked_length, config.vocab_size)
        logits_masked = F.softmax(logits_gen[masked_bool].view(-1, self.discriminator.electra.config.vocab_size), dim=1)
        # replaced tokens are set with logits
        tokens_replaced = logits_masked.multinomial(num_samples=1, replacement=True).view(-1)
        input_ids_disc = labels.clone()
        input_ids_disc[masked_bool] = tokens_replaced
        labels_disc = (input_ids_disc != labels).to(torch.long)

        outputs_disc = self.discriminator(
            input_ids=input_ids_disc, attention_mask=attention_mask, token_type_ids=token_type_ids, 
            position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, 
            labels=labels_disc, output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        # labels (torch.LongTensor of shape (batch_size, sequence_length), optional) –
        # Labels for computing the ELECTRA loss. Input should be a sequence of tokens (see input_ids docstring) Indices should be in [0, 1]:
        # - 0 indicates the token is an original token,
        # - 1 indicates the token was replaced.

        if not return_dict:
            loss_disc = outputs_disc[0]
            total_loss = self.loss_weights[0] * loss_gen + self.loss_weights[1] * loss_disc
            return ((total_loss,) + outputs_disc) if total_loss is not None else outputs_disc

        loss_disc = outputs_disc.loss
        total_loss = self.loss_weights[0] * loss_gen + self.loss_weights[1] * loss_disc
        return ElectraForPreTrainingOutput(
            loss=total_loss,
            logits=outputs_disc.logits,
            hidden_states=outputs_disc.hidden_states,
            attentions=outputs_disc.attentions,
        )
