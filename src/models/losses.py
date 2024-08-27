import torch.nn as nn


class GPTLoss(nn.Module):
    def __init__(self):
        super(GPTLoss, self).__init__()
        self.ccl = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        return self.ccl(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
        )
