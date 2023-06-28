from transformers import AutoModel, AutoModelForSequenceClassification
import torch
from torch.nn import functional as F


class AutoCrossEncoder(torch.nn.Module):
    """ reranker model (cross-attention model) """
    def __init__(self, model_name_or_path):
        super(AutoCrossEncoder, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path,num_labels=1)
                
    def forward(self, input_ids=None,
                      attention_mask=None,
                      token_type_ids=None,
                      output_attentions=False,
                      output_hidden_states=True,
                      return_dict=False,
                      labels=None):

        output = self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          output_attentions=output_attentions,
                          output_hidden_states=output_hidden_states,
                          return_dict=return_dict,
                          )
        logits = output[0]
        
        if labels != None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            return loss
        else:
            return torch.sigmoid(logits)