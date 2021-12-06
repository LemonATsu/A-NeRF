import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .utils.skeleton_utils import SMPLSkeleton

# Positional encoding
class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']

        out_dim = 0
        if self.kwargs['include_input']:
            #embed_fns.append(lambda x, **kwargs: x)
            embed_fns.append(lambda x, **kwargs: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq.item()))
                out_dim += d

        self.freq_bands = freq_bands
        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs, **kwargs):
        return self._embed(inputs, **kwargs)

    def _embed(self, inputs, **kwargs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1), None

    def update_threshold(self, *args, **kwargs):
        pass

    def update_tau(self, *args, **kwargs):
        pass

    def update_alpha(self, *args, **kwargs):
        pass

    def get_tau(self):
        return 0.0


class CutoffEmbedder(Embedder):

    def __init__(self, cutoff_dist=500*0.00035,
                 std=0.1, normalize=False, dist_inputs=False,
                 cutoff_inputs=False, opt_cutoff=False,
                 cutoff_dim=24, freq_schedule=False, init_alpha=0.,
                 cut_to_cutoff=False, shift_inputs=False, **kwargs):
        """
        cutoff_inputs: apply cutoff to the none-encoded input as well
        # TODO: poor naming ...
        dist_inputs: whether to use an extra 'dist' input to calculate the cutoff
        """

        super().__init__(**kwargs)
        self.cutoff_dist = cutoff_dist
        self.std = std
        self.normalize = normalize
        self.dist_inputs = dist_inputs
        self.cutoff_inputs = cutoff_inputs
        self.cut_to_cutoff = cut_to_cutoff
        self.shift_inputs = shift_inputs

        self.opt_cutoff = opt_cutoff
        self.cutoff_dim = cutoff_dim
        self.detach_idx = None

        self.freq_bands = self.freq_bands.view(1, -1, 1).expand(-1, -1, cutoff_dim)
        self.freq_k = torch.log2(self.freq_bands)[..., :len(self.kwargs["periodic_fns"])]
        self.freq_schedule = freq_schedule

        self.cutoff_dist = nn.Parameter(torch.ones(cutoff_dim) * self.cutoff_dist,
                                            requires_grad=False)

        self.init_tau = 20.
        self.register_buffer('tau', torch.tensor(self.init_tau))

        if self.freq_schedule:
            self.init_alpha = init_alpha
            self.register_buffer('sched_alpha', torch.tensor(self.init_alpha))
        print(f"Normalization: {self.normalize} opt_cut :{self.opt_cutoff}")

    def get_tau(self):
        return self.tau.item()

    def get_cutoff_dist(self):
        # 4, 5: knees
        # 7, 8: ankles
        # 10,11: foot
        return self.cutoff_dist

    def _embed(self, inputs, dists=None, masks=None):

        skipped = None
        # TODO: hahaha refactor this ..

        if self.dist_inputs:
            # assume that we have 1-to-1 correspondence between inputs and dists!
            input_size, dist_size = inputs.size(-1), dists.size(-1)
            if input_size >= dist_size:
                expand = input_size // dist_size
                dists = dists[..., None].expand(*dists.shape, expand).flatten(start_dim=-2)
                inputs_freq = self.freq_bands[..., None].expand(-1, -1, -1, expand).flatten(start_dim=-2).to(inputs.device) * inputs[..., None, :]
            else:
                raise NotImplementedError("This should not occur! check why embedding is broken")
        else:
            dists = inputs
            if self.cut_to_cutoff:
                inputs = self.get_cutoff_dist() - inputs
            if self.shift_inputs:
                cutoff_dist = self.get_cutoff_dist()
                # so that the frequencies, after applying cutoff, span [-1, 1]
                shifted = inputs * (2. / cutoff_dist ) - 1.
                #shifted = inputs * 2. - 0.5
                inputs_freq = self.freq_bands.to(inputs.device) * shifted[..., None, :]
            else:
                inputs_freq = self.freq_bands.to(inputs.device) * inputs[..., None, :]

        # compute cutoff weights
        cutoff_dist = self.get_cutoff_dist()
        if not self.dist_inputs:
            v = self.tau * (dists - cutoff_dist)
        else:
            v = self.tau * (dists - cutoff_dist[:, None].expand(-1, expand).flatten(start_dim=-2))
        v = v[..., None, :]
        w = 1. - torch.sigmoid(v)

        # (B, NF, NJ): shaping like the old version for backward consistency
        # stack the sin/cosine encoding and apply weights
        embedded = torch.stack([torch.sin(inputs_freq),
                                torch.cos(inputs_freq)], dim=-2).flatten(start_dim=-3, end_dim=-2)
        embedded = embedded * self.get_schedule_w()
        if self.kwargs['include_input'] and self.cutoff_inputs:
            embedded = torch.cat([inputs[..., None, :], embedded], dim=-2)
            embedded = (embedded * w)
        elif self.kwargs['include_input']:
            embedded = (embedded * w)
            embedded = torch.cat([inputs[..., None, :], embedded], dim=-2)
        else:
            embedded = (embedded * w)

        if self.normalize:
            # TODO: currently assume last dimension is 3, and weights are the same for all 3 dims!
            w_sh = w.shape
            e_sh = embedded.shape
            dists = dists.view(-1, 24, 3)
            # mask out things that are close to zero, and normalize the rest.
            is_zero = torch.isclose(w.view(-1, 3)[:, :1], torch.tensor(0.), atol=1e-6).float()
            embedded = F.normalize(embedded.view(-1, 3), p=2, dim=-1)
            embedded = torch.lerp(embedded, torch.zeros_like(embedded), is_zero)
            embedded = embedded.reshape(*e_sh)

        embedded = embedded.flatten(start_dim=-2)

        return embedded, w

    def update_threshold(self, global_step, tau_step, tau_rate,
                         alpha_step, alpha_target):
        self.update_tau(global_step, tau_step, tau_rate)
        self.update_alpha(global_step, alpha_step, alpha_target)

    def update_tau(self, global_step, step, rate):
        # TODO: makes the starting value not fixed!
        self.tau =  (self.init_tau * torch.ones_like(self.tau) * rate**(global_step / float(step * 1000))).clamp(max=2000.)

    def update_alpha(self, global_step, step, target=None):
        if not self.freq_schedule:
            return
        if target is None:
            target = torch.max(self.freq_k)
        self.sched_alpha = torch.tensor(self.init_alpha + (target - self.init_alpha) * global_step / float(step * 1000))

    def get_schedule_w(self):
        if not self.freq_schedule:
            return 1.
        diff = torch.clamp(self.sched_alpha - self.freq_k.to(self.sched_alpha.device), 0, 1)
        w = (0.5 * (1. - torch.cos(np.pi * diff))).view(1, -1, 1)
        return w

def get_embedder(multires, i=0, input_dims=3,
                 cutoff_kwargs={"cutoff": False},
                 skel_type=SMPLSkeleton,
                 kc=False):
    if i == -1:
        return nn.Identity(), input_dims

    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
                'skel_type': skel_type,
    }

    if cutoff_kwargs["cutoff"]:
        embed_kwargs.update(cutoff_kwargs)
        embedder_class = CutoffEmbedder
    else:
        embedder_class = Embedder
    print(f"Embedder class: {embedder_class}")

    embedder_obj = embedder_class(**embed_kwargs)
    return embedder_obj, embedder_obj.out_dim


def load_ckpt_from_path(ray_caster, optimizer, ckpt_path,
                        finetune=False):
    ckpt = torch.load(ckpt_path)

    global_step = ckpt["global_step"]
    ray_caster.load_state_dict(ckpt)
    if optimizer is not None and not finetune:
        print("load optimizer from ckpt")
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    return global_step, ray_caster, optimizer, ckpt

