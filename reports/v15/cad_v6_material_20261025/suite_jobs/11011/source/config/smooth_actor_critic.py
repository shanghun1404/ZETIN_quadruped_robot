from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, Memory


def _mlp(input_dim: int, hidden_dims: list[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for hidden in hidden_dims:
        layers.extend((nn.Linear(current, hidden), nn.ELU()))
        current = hidden
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class SmoothActorCritic(nn.Module):
    is_recurrent = True

    def __init__(
        self,
        obs: dict[str, torch.Tensor],
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = True,
        critic_obs_normalization: bool = True,
        init_noise_std: float = 0.35,
        noise_std_type: str = "scalar",
        **_: object,
    ) -> None:
        super().__init__()
        expected = {"proprio": 46, "terrain": 187, "command": 5, "critic": 246}
        for name, width in expected.items():
            if name not in obs or obs[name].shape[-1] != width:
                actual = None if name not in obs else obs[name].shape[-1]
                raise ValueError(f"{name} observation width must be {width}, got {actual}")

        self.obs_groups = obs_groups
        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        self.proprio_normalizer = EmpiricalNormalization(46) if actor_obs_normalization else nn.Identity()
        self.terrain_normalizer = EmpiricalNormalization(187) if actor_obs_normalization else nn.Identity()
        self.command_normalizer = EmpiricalNormalization(5) if actor_obs_normalization else nn.Identity()
        self.critic_normalizer = EmpiricalNormalization(246) if critic_obs_normalization else nn.Identity()

        self.proprio_encoder = nn.Sequential(nn.Linear(46, 64), nn.ELU())
        self.terrain_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 6, 128),
            nn.ELU(),
            nn.Linear(128, 64),
        )
        self.command_encoder = nn.Sequential(nn.Linear(5, 32), nn.ELU())
        self.memory_a = Memory(160, type="gru", num_layers=1, hidden_size=128)
        self.memory_c = Memory(246, type="gru", num_layers=1, hidden_size=128)
        self.actor = _mlp(128, [256, 256, 128], num_actions)
        self.critic = _mlp(128, [512, 256, 128], 1)

        self.noise_std_type = noise_std_type
        self.min_std = 0.05
        self.max_std = 0.80
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unsupported noise_std_type: {noise_std_type}")
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    @staticmethod
    def _encode_vector(encoder: nn.Module, value: torch.Tensor) -> torch.Tensor:
        leading = value.shape[:-1]
        encoded = encoder(value.reshape(-1, value.shape[-1]))
        return encoded.reshape(*leading, encoded.shape[-1])

    def encode_terrain(self, terrain: torch.Tensor) -> torch.Tensor:
        leading = terrain.shape[:-1]
        encoded = self.terrain_encoder(terrain.reshape(-1, 1, 17, 11))
        return encoded.reshape(*leading, encoded.shape[-1])

    def _actor_features(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        proprio = self.proprio_normalizer(obs["proprio"])
        terrain = self.terrain_normalizer(obs["terrain"])
        command = self.command_normalizer(obs["command"])
        return torch.cat(
            (
                self._encode_vector(self.proprio_encoder, proprio),
                self.encode_terrain(terrain),
                self._encode_vector(self.command_encoder, command),
            ),
            dim=-1,
        )

    def _actor_mean(
        self,
        obs: dict[str, torch.Tensor],
        masks: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        recurrent = self.memory_a(self._actor_features(obs), masks, hidden_states).squeeze(0)
        return torch.tanh(self.actor(recurrent))

    def update_distribution(
        self,
        obs: dict[str, torch.Tensor],
        masks: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
    ) -> None:
        mean = self._actor_mean(obs, masks, hidden_states)
        if self.noise_std_type == "scalar":
            std = torch.clamp(self.std, self.min_std, self.max_std).expand_as(mean)
        else:
            log_std = torch.clamp(self.log_std, math.log(self.min_std), math.log(self.max_std))
            std = torch.clamp(torch.exp(log_std), self.min_std, self.max_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(
        self,
        obs: dict[str, torch.Tensor],
        masks: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.update_distribution(obs, masks, hidden_states)
        return self.distribution.sample()

    def act_inference(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._actor_mean(obs)

    def evaluate(
        self,
        obs: dict[str, torch.Tensor],
        masks: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        critic_obs = self.critic_normalizer(obs["critic"])
        recurrent = self.memory_c(critic_obs, masks, hidden_states).squeeze(0)
        return self.critic(recurrent)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: dict[str, torch.Tensor]) -> None:
        if self.actor_obs_normalization:
            self.proprio_normalizer.update(obs["proprio"].reshape(-1, 46))
            self.terrain_normalizer.update(obs["terrain"].reshape(-1, 187))
            self.command_normalizer.update(obs["command"].reshape(-1, 5))
        if self.critic_obs_normalization:
            self.critic_normalizer.update(obs["critic"].reshape(-1, 246))

    def reset(self, dones: torch.Tensor | None = None) -> None:
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states

    def forward(self) -> None:
        raise NotImplementedError

    def load_state_dict(self, state_dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True
