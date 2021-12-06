import torch
import torch.nn as nn

class Optcodes(nn.Module):

    def __init__(self, n_codes, code_ch, idx_map=None, transform_code=False, mean=None, std=None):
        super().__init__()
        self.n_codes = n_codes
        self.code_ch = code_ch
        self.codes = nn.Embedding(n_codes, code_ch)
        self.transform_code = transform_code
        self.idx_map = None
        if idx_map is not None:
            self.idx_map = torch.LongTensor(idx_map)
        self.init_parameters(mean, std)

    def forward(self, idx, t=None, *args, **kwargs):

        if self.idx_map is not None:
            idx = self.idx_map[idx.long()].to(idx.device)
        if not self.training and idx.max() < 0:
            codes = self.codes.weight.mean(0, keepdims=True).expand(len(idx), -1)
        else:
            if idx.shape[-1] != 1:
                codes = self.codes(idx[..., :2].long()).squeeze(1)
                w = idx[..., 2]
                # interpolate given mixing weights
                codes = torch.lerp(codes[..., 0, :], codes[..., 1, :], w[..., None])
            else:
                codes = self.codes(idx.long()).squeeze(1)

        if self.transform_code:
            codes = codes.view(t.shape[0], 4, -1).flatten(start_dim=-2)
        return codes

    def init_parameters(self, mean=None, std=None):
        if mean is None:
            nn.init.xavier_normal_(self.codes.weight)
            return

        if std > 0.:
            nn.init.normal_(self.codes.weight, mean=mean, std=std)
        else:
            nn.init.constant_(self.codes.weight, mean)

