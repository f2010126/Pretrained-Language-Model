import torch
import torch.nn as nn
from transformers import BertModel
from abc import ABC, abstractmethod
from collections import defaultdict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = defaultdict(int)
        self.avg = defaultdict(float)
        self.sum = defaultdict(int)
        self.count = defaultdict(int)

    def update(self, n=1, **val):
        for k in val:
            self.val[k] = val[k]
            self.sum[k] += val[k] * n
            self.count[k] += n
            self.avg[k] = self.sum[k] / self.count[k]


def get_error(output, target):
    pred = output.argmax(dim=1)
    correct = pred.eq(target).float().sum()
    return float((1. - correct / output.size(0)) * 100.)

def get_device(model: torch.nn.Module):
    return next(model.parameters()).device

class ProbeNetwork(ABC, nn.Module):
    """Abstract class that all probe networks should inherit from.

    This is a standard torch.nn.Module but needs to expose a classifier property that returns the final classicifation
    module (e.g., the last fully connected layer).
    """

    @property
    @abstractmethod
    def classifier(self):
        raise NotImplementedError("Override the classifier property to return the submodules of the network that"
                                  " should be interpreted as the classifier")

    @classifier.setter
    @abstractmethod
    def classifier(self, val):
        raise NotImplementedError("Override the classifier setter to set the submodules of the network that"
                                  " should be interpreted as the classifier")


# Adapt BertModel as ProbeNetwork
class BertProbeNetwork(BertModel, ProbeNetwork):
   
    def __init__(self):
        super(BertProbeNetwork, self).__init__.from_pretrained('dbmdz/bert-base-german-uncased')
    
    @property
    def classifier(self):
        return self.cls

def get_bert_probe_network(pretrained=True):
    # Load pre-trained BERT model
    bert_model = BertModel.from_pretrained('dbmdz/bert-base-german-uncased')
    probe_network = BertProbeNetwork(bert_model)
    return probe_network


