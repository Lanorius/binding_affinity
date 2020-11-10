from abc import ABC

import torch
import torch.nn.functional as f
import torch.nn as nn

# I believe that we should leave this file the way it is.
# If we finish everything else really fast, this is a place to experiment.


class PcNet(nn.Module, ABC):
    def __init__(self, input_size_prot=1024, input_size_comp=196, hidden_size_prot=32):
        super(PcNet, self).__init__()
        self.fc_prot = nn.Linear(input_size_prot, hidden_size_prot)
        self.fc_lin1 = nn.Linear(hidden_size_prot+input_size_comp, 1024)
        self.fc_drop1 = nn.Dropout(0.1)
        self.fc_lin2 = nn.Linear(1024, 1024)
        self.fc_drop2 = nn.Dropout(0.1)
        self.fc_lin3 = nn.Linear(1024, 512)
        self.fc_lin4 = nn.Linear(512, 1)

    def forward(self, x):
        out = f.relu(self.fc_prot(x[0]))
        out = torch.cat((out, x[1]), dim=1)
        out = self.fc_drop1(f.relu(self.fc_lin1(out)))
        out = self.fc_drop2(f.relu(self.fc_lin2(out)))
        out = f.relu(self.fc_lin3(out))
        out1 = f.relu(self.fc_lin4(out))

        return out1
