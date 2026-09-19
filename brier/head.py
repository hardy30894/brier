"""A typed head over hidden states. It starts as the identity on the LM-head scores and learns
two corrections: an input-conditional temperature from the state's representation, and a
per-candidate residual from the candidate's representation.

    score_i = lm_i * exp(tau . h_state) + w . h_cand_i + b

All parameters start at zero, so an untrained head reproduces the plain scorer exactly."""
import torch


class TypedHead(torch.nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.tau = torch.nn.Parameter(torch.zeros(d))
        self.w = torch.nn.Parameter(torch.zeros(d))
        self.b = torch.nn.Parameter(torch.zeros(1))

    def forward(self, lm_scores, h_state, h_cand):
        scale = torch.exp((h_state.float() @ self.tau).clamp(-3, 3))
        return lm_scores * scale + h_cand.float() @ self.w + self.b
