import torch
import torch.nn.functional as F


class NTXent(torch.nn.Module):
    def __init__(self, temperature, batch_size=None, device=None):
        super().__init__()
        self.temperature = temperature
        self._batch_size = batch_size
        if batch_size is not None:
            assert device is not None
            self._self_mask, self._pos_mask = self._build_masks(batch_size, device)
        self.enforce_batch_size()

    def _build_masks(self, N, device):
        # To select the similarity of each vector with itself
        self_mask = torch.eye(2 * N, dtype=bool).to(device)
        # To select the similarity of each vector with its positive pair
        pos_mask = torch.roll(self_mask, N, 0)
        return self_mask, pos_mask

    def enforce_batch_size(self, enforce=True):
        # If a batch size was not spcified, we have nothing to enforce
        if self._batch_size is None:
            enforce = False
        self._enforce_batch_size = enforce

    def forward(self, z):
        # Make sure z_i and z_j have the same dims (N * projection_dim)
        assert z[0].size() == z[1].size()

        # The size of the received batches
        N = z[0].size(0)

        # If a batch_size was specified at build time, ensure that it matches N
        if self._enforce_batch_size:
            assert N == self._batch_size
            self_mask, pos_mask = self._self_mask, self._pos_mask
        else:
            self_mask, pos_mask = self._build_masks(N, z[0].device)

        # Stack vertically -> 2N * projection_dim
        z = torch.cat(z, dim=0)

        # 2N * 2N
        sim = F.cosine_similarity(z.unsqueeze(0), z.unsqueeze(1), dim=-1)
        sim /= self.temperature

        # We do not consider the similarity of a feature vector with itself
        sim.masked_fill_(self_mask, float("-inf"))

        # Select the similarities of the positive pairs
        positives_sim = sim.masked_select(pos_mask)

        # Compute the loss for each sample
        losses = -positives_sim + torch.logsumexp(sim, 1)

        # Average across the batch
        return losses.mean()
