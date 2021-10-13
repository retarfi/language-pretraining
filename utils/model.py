from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from transformers import (
    ElectraConfig, 
    ElectraForMaskedLM, 
    ElectraForPreTraining, 
    ElectraPreTrainedModel,
    Trainer
)
from transformers.models.electra.modeling_electra import ElectraForPreTrainingOutput
from transformers.file_utils import ModelOutput

@dataclass
class MyElectraForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.ElectraForPreTraining`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss of the ELECTRA objective.
        gen_loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Generator loss of the ELECTRA objective.
        disc_loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Discriminator loss of the ELECTRA objective.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`):
            Prediction scores of the head (scores for each token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    gen_loss: Optional[torch.FloatTensor] = None
    disc_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class ElectraForPretrainingModel(ElectraPreTrainedModel):
    def __init__(self, config_generator, config_discriminator, loss_weights=(1.0,50.0)):
        super().__init__(config_discriminator)

        self.generator = ElectraForMaskedLM(config_generator)
        self.discriminator = ElectraForPreTraining(config_discriminator)
        # weight sharing
        self.discriminator.electra.embeddings = self.generator.electra.embeddings
        self.generator.generator_lm_head.weight = self.generator.electra.embeddings.word_embeddings.weight
        self.init_weights()
        self.loss_weights = loss_weights
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(0.,1.)
    
    def to(self, *args, **kwargs):
        "Also set dtype and device of contained gumbel distribution if needed"
        return_object = super().to(*args, **kwargs)
        device, dtype = self.generator.device, torch.float32
        # https://github.com/pytorch/pytorch/issues/41663
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(torch.tensor(0., device=device, dtype=dtype), torch.tensor(1., device=device, dtype=dtype))
        return return_object

    def forward(
            self, 
            input_ids, labels, attention_mask=None, token_type_ids=None, 
            position_ids=None,
            output_attentions=None, 
            output_hidden_states=None, return_dict=None
        ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs_gen = self.generator(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
            position_ids=position_ids,
            labels=labels, output_attentions=False, 
            output_hidden_states=False, return_dict=True
        )

        loss_gen = outputs_gen.loss # (1,)
        if torch.isnan(loss_gen):
            raise ValueError('output_gen is NaN')
        logits_gen = outputs_gen.logits # (batch_size, seq_length, config.vocab_size)
        with torch.no_grad():
            masked_bool = (labels != -100)
            # ids_answer = input_ids.clone()
            # ids_answer[masked_bool] = labels[masked_bool]

            logits = logits_gen[masked_bool]
            gumbel = self.gumbel_dist.sample(logits.shape)
            tokens_replaced = (logits + gumbel).argmax(dim=-1)
            input_ids_disc = input_ids.clone()
            input_ids_disc[masked_bool] = tokens_replaced
            labels_disc = torch.zeros(labels.shape, dtype=torch.long, device=labels.device)
            labels_disc[masked_bool] = (tokens_replaced != labels[masked_bool]).to(torch.long)

        outputs_disc = self.discriminator(
            input_ids=input_ids_disc, attention_mask=attention_mask, token_type_ids=token_type_ids, 
            position_ids=position_ids,
            labels=labels_disc, output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states, return_dict=return_dict
        )

        if not return_dict:
            loss_disc = outputs_disc[0]
            total_loss = self.loss_weights[0] * loss_gen + self.loss_weights[1] * loss_disc
            return ((total_loss, loss_gen.detach(), loss_disc.detach()) + outputs_disc[1:]) if total_loss is not None else outputs_disc

        loss_disc = outputs_disc.loss
        if torch.isnan(loss_disc):
            raise ValueError('loss_disc is NaN')
        total_loss = self.loss_weights[0] * loss_gen + self.loss_weights[1] * loss_disc
        return MyElectraForPreTrainingOutput(
            loss=total_loss,
            gen_loss=loss_gen.detach(),
            disc_loss=loss_disc.detach(),
            logits=outputs_disc.logits,
            hidden_states=outputs_disc.hidden_states,
            attentions=outputs_disc.attentions,
        )
