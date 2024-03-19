import torch
from gem_cnn.nn.gem_res_net_block import GemResNetBlock
import torch.nn as nn
from torch.nn import Sequential
import torch.nn.functional as F

class GEMCNN(torch.nn.Module):
    def __init__(self, nverts, nlayers, channels=16, convs=3, n_rings=2, max_order=2, len_features=32):
        """
        GEM-CNN Network used for SIRE.
        nverts: number of vertices on each surface of the sphere
        nlayers: number of scales (needed for reshaping the output of SIRE)
        """
        super(GEMCNN, self).__init__()

        # onion structure
        self.nverts = nverts * nlayers
        self.nlayers = nlayers
        self.len_features = len_features

        kwargs = dict(
            n_rings=n_rings,
            band_limit=max_order,
            num_samples=7,
            checkpoint=False,
            batch=100000,
            batch_norm=False,
        )

        model = [GemResNetBlock(self.len_features, channels, 0, max_order, **kwargs)]
        for i in range(convs - 2):
            model += [
                GemResNetBlock(channels, channels, max_order, max_order, **kwargs)
            ]
        model += [GemResNetBlock(channels, channels, max_order, 0, **kwargs)]

        self.model = Sequential(*model)

        # Dense final layer
        self.lin1 = nn.Linear(channels, 1)

    def forward(self, data, multiscale=True):
        attr0 = (data.edge_index, data.precomp, data.connection)

        x = data.features.reshape(-1, self.len_features, 1).float()

        for layer in self.model:
            x = layer(x, *attr0)

        # Take trivial feature
        x = x[:, :, 0]

        x = F.relu(self.lin1(x)).squeeze()
        # reshape to Onion part!!
        if multiscale:  # otherwise, obtain the raw single scale output of the network
            x = torch.cat(
                [
                    x[i * self.nverts : (i + 1) * self.nverts].view(self.nlayers, -1)
                    for i in range(len(x) // self.nverts)
                ],
                dim=1,
            ).transpose(0, 1)
        return x