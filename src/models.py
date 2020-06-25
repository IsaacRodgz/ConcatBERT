import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel

class AverageBERTModel(nn.Module):

    def __init__(self, hyp_params):

        super(AverageBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(hyp_params.bert_model)
        self.drop1 = nn.Dropout(p=hyp_params.mlp_dropout)
        self.linear1 = nn.Linear(self.bert.config.hidden_size+hyp_params.image_feature_size, 512)
        self.drop2 = nn.Dropout(p=hyp_params.mlp_dropout)
        self.linear2 = nn.Linear(512, hyp_params.output_dim)

    def forward(self, input_ids, attention_mask, feature_images):

        last_hidden, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        mean_hidden = torch.mean(last_hidden, dim = 1)

        x = torch.cat((mean_hidden, feature_images), dim=1)
        x = self.drop1(x)
        x = self.linear1(x)
        x = self.drop2(x)

        return self.linear2(x)


class ConcatBERTModel(nn.Module):

    def __init__(self, hyp_params):

        super(ConcatBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(hyp_params.bert_model)
        self.drop1 = nn.Dropout(p=hyp_params.mlp_dropout)
        self.linear1 = nn.Linear(self.bert.config.hidden_size+hyp_params.image_feature_size, 512)
        self.drop2 = nn.Dropout(p=hyp_params.mlp_dropout)
        self.linear2 = nn.Linear(512, hyp_params.output_dim)

    def forward(self, input_ids, attention_mask, feature_images):

        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        x = torch.cat((pooled_output, feature_images), dim=1)
        x = self.drop1(x)
        x = self.linear1(x)
        x = self.drop2(x)

        return self.linear2(x)
