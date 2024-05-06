import itertools
import torch


class BasicOcclusion:
    def __init__(
            self,
            input_shape=(1, 3, 224, 224),
            sliding_window_shapes=(3, 28, 28)
        ):
        self.input_shape = input_shape
        self.source_shape = input_shape[1:]
        self.sliding_window_shapes = sliding_window_shapes

        # Generate mask tensors
        with torch.no_grad():
            c, h, w = self.source_shape
            mc, mh, mw = self.sliding_window_shapes
            nmc = -(-c // mc)
            nmh = -(-h // mh)
            nmw = -(-w // mw)
            self.masks = torch.ones((nmc, nmh, nmw, c, h, w))
            for i in range(0, c, mc):
                for j in range(0, h, mh):
                    for k in range(0, w, mw):
                        self.masks[
                            i // mc, j // mh, k // mw,
                            i: i + mc, j: j + mh, k: k + mw
                        ] = 0

    def attribute(self, model, input, target=None, baselines=0, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if isinstance(baselines, (int, float)):
            baselines = torch.full(self.source_shape, baselines)
        else:
            baselines = torch.as_tensor(baselines)
        model = model.to(device)
        with torch.no_grad():
            output = model(input.to(device))
        if target is None:
            target = output.argmax(dim=1).item()
        logit = output[0, target]
        input_group = input * self.masks + baselines * (1 - self.masks)
        input_group = input_group.view(-1, *self.source_shape)
        output_group = torch.zeros(input_group.size(0), output.shape[-1], device=device)
        with torch.no_grad():
            # Batch input_group into smaller groups to avoid memory error
            for i in range(0, input_group.size(0), 100):
                output = model(input_group[i: i + 100].to(device))
                output_group[i: i + 100] = output
        sensitivity = output_group[:, target].view(*self.masks.shape[:-3])
        return logit - sensitivity


class CorrelatedOcclusion(BasicOcclusion):
    def __init__(
            self,
            input_shape=(1, 3, 224, 224),
            sliding_window_shapes=(3, 28, 28),
            order=2
        ):
        self.input_shape = input_shape
        self.source_shape = input_shape[1:]
        self.sliding_window_shapes = sliding_window_shapes
        self.order = order

        # Generate mask tensors
        with torch.no_grad():
            c, h, w = self.source_shape
            mc, mh, mw = self.sliding_window_shapes
            nmc = -(-c // mc)
            nmh = -(-h // mh)
            nmw = -(-w // mw)
            self.masks_ = torch.ones((nmc, nmh, nmw, c, h, w))
            for i in range(0, c, mc):
                for j in range(0, h, mh):
                    for k in range(0, w, mw):
                        self.masks_[
                            i // mc, j // mh, k // mw,
                            i: i + mc, j: j + mh, k: k + mw
                        ] = 0
            self.masks_idx_ = list(itertools.product(range(nmc), range(nmh), range(nmw)))
            self.masks_idx = list(itertools.product(self.masks_idx_, repeat=self.order))
            masks_dim = (nmc, nmh, nmw) * self.order + (c, h, w)
            self.masks = torch.ones(masks_dim)
            for m in torch.tensor(self.masks_idx):
                for n in m:
                    self.masks[*m.view(-1)] *= self.masks_[*n]
