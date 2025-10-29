# %%
import sys

import torch

sys.path.append("..")
from src.kl_divergence import calculate_kl_score_per_token

# %%
probs_1 = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
probs_2 = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
kl_per_token, entropy_1_per_token, entropy_2_per_token, weighted_kl_per_token = (
    calculate_kl_score_per_token(
        torch.log_softmax(probs_1, dim=-1), torch.log_softmax(probs_2, dim=-1)
    )
)
print(f"{kl_per_token=}")
print(f"{weighted_kl_per_token=}")
print(f"{entropy_1_per_token=}")
print(f"{entropy_2_per_token=}")
# %%
probs_1 = torch.tensor([[0.1, 0.1, 0.1, 0.7]])
probs_2 = torch.tensor([[0.7, 0.1, 0.1, 0.1]])
kl_per_token, entropy_1_per_token, entropy_2_per_token, weighted_kl_per_token = (
    calculate_kl_score_per_token(
        torch.log_softmax(probs_1, dim=-1), torch.log_softmax(probs_2, dim=-1)
    )
)
print(f"{kl_per_token=}")
print(f"{weighted_kl_per_token=}")
print(f"{entropy_1_per_token=}")
print(f"{entropy_2_per_token=}")
# %%
probs_1 = torch.tensor([[0.6, 0.1, 0.1, 0.2]])
probs_2 = torch.tensor([[0.7, 0.1, 0.1, 0.1]])
kl_per_token, entropy_1_per_token, entropy_2_per_token, weighted_kl_per_token = (
    calculate_kl_score_per_token(
        torch.log_softmax(probs_1, dim=-1), torch.log_softmax(probs_2, dim=-1)
    )
)
print(f"{kl_per_token=}")
print(f"{weighted_kl_per_token=}")
print(f"{entropy_1_per_token=}")
print(f"{entropy_2_per_token=}")

# %%
