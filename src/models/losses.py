import torch.nn as nn


class GPTLoss(nn.Module):
    def __init__(self):
        super(GPTLoss, self).__init__()
        self.ccl = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        return self.ccl(inputs, targets)
