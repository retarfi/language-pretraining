import torch
import torch.nn as nn
from transformers import (
    ElectraConfig, 
    ElectraForMaskedLM, 
    PreTrainedModel,
    ElectraForPreTraining, 
    PreTrainedModel,
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
        # self.init_weights()
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
        input_ids=None, attention_mask=None, token_type_ids=None, 
        position_ids=None, head_mask=None, inputs_embeds=None, 
        labels=None, output_attentions=None, 
        output_hidden_states=None, return_dict=None
    
    ):
        outputs_gen = self.generator(
            input_ids=input_ids, attention_mask=None, token_type_ids=token_type_ids, 
            position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, 
            labels=labels, output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        # return: (loss, logits, hidden_states, attentions)
        # >>> inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")
        # >>> labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
        # >>> outputs = model(**inputs, labels=labels)

        loss_gen = outputs_gen.loss # shape: (1,)
        if labels is not None:
            # Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            logits_gen = outputs_gen.logits # shape: (batch_size, sequence_length, config.vocab_size)
            predict_ids = torch.argmax(logits_gen, dim=2) + 1
            labels_disc = (predict_ids != labels).to(torch.long)
        else:
            labels_disc = None

        outputs_disc = self.discriminator(
            input_ids=predict_ids, attention_mask=None, token_type_ids=token_type_ids, 
            position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, 
            labels=labels_disc, output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        # labels (torch.LongTensor of shape (batch_size, sequence_length), optional) –
        # Labels for computing the ELECTRA loss. Input should be a sequence of tokens (see input_ids docstring) Indices should be in [0, 1]:
        # - 0 indicates the token is an original token,
        # - 1 indicates the token was replaced.

        if labels is not None:
            loss_disc = outputs_disc.loss
            total_loss = self.loss_weights[0] * loss_gen + self.loss_weights[1] * loss_disc
        else:
            total_loss = None

        return ElectraForPreTrainingOutput(
            loss=total_loss,
            logits=outputs_disc.logits,
            hidden_states=outputs_disc.hidden_states,
            attentions=outputs_disc.attentions,
        )
