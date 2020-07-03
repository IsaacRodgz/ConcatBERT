import math
import torch
from torch import nn
import torch.nn.functional as F


class GatedMultimodalLayer(nn.Module):
    """ Gated Multimodal Layer based on 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) """
    def __init__(self, size_in1, size_in2, size_out):
        super().__init__()
        self.size_in1, self.size_in2, self.size_out = size_in1, size_in2, size_out

        # Weights hidden state modality 1
        weights_hidden1 = torch.Tensor(size_out, size_in1)
        self.weights_hidden1 = nn.Parameter(weights_hidden1)

        # Weights hidden state modality 2
        weights_hidden2 = torch.Tensor(size_out, size_in2)
        self.weights_hidden2 = nn.Parameter(weights_hidden2)

        # Weight for sigmoid
        weight_sigmoid = torch.Tensor(size_out*2)
        self.weight_sigmoid = nn.Parameter(weight_sigmoid)

        # initialize weights
        nn.init.kaiming_uniform_(self.weights_hidden1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weights_hidden2, a=math.sqrt(5))

        # Activation functions
        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x1, x2):
        h1 = self.tanh_f(torch.mm(x1, self.weights_hidden1.t()))
        h2 = self.tanh_f(torch.mm(x2, self.weights_hidden2.t()))
        x = torch.cat((h1, h2), dim=1)
        z = self.sigmoid_f(torch.matmul(x, self.weight_sigmoid.t()))

        return z.view(z.size()[0],1)*h1 + (1-z).view(z.size()[0],1)*h2


class AverageBERTModel(nn.Module):

    def __init__(self, hyp_params):

        super(AverageBERTModel, self).__init__()
        self.linear1 = nn.Linear(hyp_params.bert_hidden_size+hyp_params.image_feature_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(p=hyp_params.mlp_dropout)
        self.linear2 = nn.Linear(512, hyp_params.output_dim)

    def forward(self, last_hidden, pooled_output, feature_images):

        mean_hidden = torch.mean(last_hidden, dim = 1)

        x = torch.cat((mean_hidden, feature_images), dim=1)
        x = self.drop1(x)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.drop1(x)

        return self.linear2(x)


class ConcatBERTModel(nn.Module):

    def __init__(self, hyp_params):

        super(ConcatBERTModel, self).__init__()
        self.linear1 = nn.Linear(hyp_params.bert_hidden_size+hyp_params.image_feature_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(p=hyp_params.mlp_dropout)
        self.linear2 = nn.Linear(512, hyp_params.output_dim)

    def forward(self, last_hidden, pooled_output, feature_images):

        x = torch.cat((pooled_output, feature_images), dim=1)
        x = self.drop1(x)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.drop1(x)

        return self.linear2(x)


class GatedAverageBERTModel(nn.Module):

    def __init__(self, hyp_params):

        super(GatedAverageBERTModel, self).__init__()
        self.gated_linear1 = GatedMultimodalLayer(hyp_params.bert_hidden_size, hyp_params.image_feature_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(p=hyp_params.mlp_dropout)
        self.linear1 = nn.Linear(512, hyp_params.output_dim)

    def forward(self, last_hidden, pooled_output, feature_images):

        mean_hidden = torch.mean(last_hidden, dim = 1)

        x = self.gated_linear1(mean_hidden, feature_images)
        x = self.bn1(x)
        x = self.drop1(x)

        return self.linear1(x)
