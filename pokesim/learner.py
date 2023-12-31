import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy
from typing import Any, List, Mapping

from pokesim.constants import _NUM_PLAYERS
from pokesim.model.main import Model
from pokesim.structs import Batch
from pokesim.utils import preprocess, optimized_forward
from pokesim.rl_utils import EntropySchedule, SGDTowardsModel, v_trace, _player_others
from pokesim.config import RNaDConfig


def get_loss_v_(
    v_n: torch.Tensor,
    v_target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    loss_v = torch.unsqueeze(mask, dim=-1) * (v_n - torch.detach(v_target)) ** 2
    normalization = torch.sum(mask)
    loss_v = torch.sum(loss_v) / torch.clamp(normalization, min=1)
    return loss_v


def get_loss_v(
    v_list: List[torch.Tensor],
    v_target_list: List[torch.Tensor],
    mask_list: List[torch.Tensor],
) -> torch.Tensor:
    """Define the loss function for the critic."""
    loss_v_list = []
    for v_n, v_target, mask in zip(v_list, v_target_list, mask_list):
        loss_v = get_loss_v_(v_n, v_target, mask)
        loss_v_list.append(loss_v)
    return sum(loss_v_list)


def apply_force_with_threshold(
    decision_outputs: torch.Tensor,
    force: torch.Tensor,
    threshold: float,
    threshold_center: torch.Tensor,
) -> torch.Tensor:
    """Apply the force with below a given threshold."""
    can_decrease = decision_outputs - threshold_center > -threshold
    can_increase = decision_outputs - threshold_center < threshold
    force_negative = torch.minimum(force, torch.tensor(0.0))
    force_positive = torch.maximum(force, torch.tensor(0.0))
    clipped_force = can_decrease * force_negative + can_increase * force_positive
    return decision_outputs * torch.detach(clipped_force)


def renormalize(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The `normalization` is the number of steps over which loss is computed."""
    loss = torch.sum(loss * mask)
    normalization = torch.sum(mask)
    return loss / torch.clamp(normalization, min=1)


def get_loss_nerd_(
    logit_pi: torch.Tensor,
    pi: torch.Tensor,
    q_vr: torch.Tensor,
    valid: torch.Tensor,
    player_ids: torch.Tensor,
    legal_actions: torch.Tensor,
    is_c: torch.Tensor,
    k: int,
    clip: float = 100,
    threshold: float = 2,
) -> torch.Tensor:
    # loss policy
    adv_pi = q_vr - torch.sum(pi * q_vr, dim=-1, keepdim=True)
    adv_pi = is_c * adv_pi  # importance sampling correction
    adv_pi = torch.clip(adv_pi, min=-clip, max=clip)
    adv_pi = torch.detach(adv_pi)

    logits = logit_pi - torch.mean(logit_pi * legal_actions, dim=-1, keepdim=True)

    threshold_center = torch.zeros_like(logits)

    nerd_loss = torch.sum(
        legal_actions
        * apply_force_with_threshold(logits, adv_pi, threshold, threshold_center),
        dim=-1,
    )
    nerd_loss = -renormalize(nerd_loss, valid * (player_ids == k))

    return nerd_loss


def get_loss_nerd(
    logit_list: List[torch.Tensor],
    policy_list: List[torch.Tensor],
    q_vr_list: List[torch.Tensor],
    valid: torch.Tensor,
    player_ids: torch.Tensor,
    legal_actions: torch.Tensor,
    importance_sampling_correction: List[torch.Tensor],
    clip: float = 100,
    threshold: float = 2,
) -> torch.Tensor:
    """Define the nerd loss."""
    loss_pi_list = []
    for k, (logit_pi, pi, q_vr, is_c) in enumerate(
        zip(logit_list, policy_list, q_vr_list, importance_sampling_correction)
    ):
        nerd_loss = get_loss_nerd_(
            logit_pi,
            pi,
            q_vr,
            valid,
            player_ids,
            legal_actions,
            is_c,
            k,
            clip,
            threshold,
        )
        loss_pi_list.append(nerd_loss)
    return sum(loss_pi_list)


def _print_params(model: nn.Module):
    params_count = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print(f"""Total Params: {params_count:,}""")


class Learner:
    def __init__(
        self, init: Mapping[str, Any] = None, config: RNaDConfig = RNaDConfig()
    ):
        self.config = config
        self._entropy_schedule = EntropySchedule(
            sizes=self.config.entropy_schedule_size,
            repeats=self.config.entropy_schedule_repeats,
        )

        # Create initial parameters.
        self.params = Model()
        if init is not None:
            self.params.load_state_dict(init)
        self.params_actor = deepcopy(self.params).share_memory()
        self.params_target = deepcopy(self.params)
        self.params_prev = deepcopy(self.params)
        self.params_prev_ = deepcopy(self.params)

        self.params.to(self.config.learner_device)
        self.params_actor.to(self.config.actor_device)
        self.params_target.to(self.config.learner_device)
        self.params_prev.to(self.config.learner_device)
        self.params_prev_.to(self.config.learner_device)

        # Parameter optimizers.
        self.optimizer = optim.Adam(
            self.params.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam.b1, self.config.adam.b2),
        )
        self.optimizer_target = SGDTowardsModel(
            self.params_target, self.params, self.config.target_network_avg
        )
        self.learner_steps = 0
        _print_params(self.params)

    def _to_torch(self, arr: np.ndarray, device: str = None):
        if device is None:
            device = self.config.learner_device
        return torch.from_numpy(arr).to(device, non_blocking=True)

    def loss(self, batch: Batch, alpha: float) -> torch.Tensor:
        obs = {k: self._to_torch(t) for k, t in preprocess(batch.raw_obs).items()}
        mask = self._to_torch(batch.legal.astype(np.bool_))

        # t_callback = lambda t: t.to("cuda", non_blocking=True)

        # pi, v, log_pi, logit = optimized_forward(
        #     self.params, {**obs, "mask": mask}, t_callback
        # )
        # with torch.no_grad():
        #     _, v_target, _, _ = optimized_forward(
        #         self.params_target, {**obs, "mask": mask}, t_callback
        #     )
        #     _, _, log_pi_prev, _ = optimized_forward(
        #         self.params_prev, {**obs, "mask": mask}, t_callback
        #     )
        #     _, _, log_pi_prev_, _ = optimized_forward(
        #         self.params_prev_, {**obs, "mask": mask}, t_callback
        #     )

        pi, v, log_pi, logit = self.params(**obs, mask=mask)
        with torch.no_grad():
            _, v_target, _, _ = self.params_target(**obs, mask=mask)
            _, _, log_pi_prev, _ = self.params_prev(**obs, mask=mask)
            _, _, log_pi_prev_, _ = self.params_prev_(**obs, mask=mask)

        # This line creates the reward transform log(pi(a|x)/pi_reg(a|x)).
        # For the stability reasons, reward changes smoothly between iterations.
        # The mixing between old and new reward transform is a convex combination
        # parametrised by alpha.
        log_policy_reg = log_pi - (alpha * log_pi_prev + (1 - alpha) * log_pi_prev_)

        v_target_list, has_played_list, v_trace_policy_target_list = [], [], []

        _rewards = batch.rewards.astype(np.float32)
        _v_target = v_target.cpu().numpy()
        _pi = pi.detach().cpu().numpy()
        _log_policy_reg = log_policy_reg.detach().cpu().numpy()

        valid = self._to_torch(batch.valid)
        player_id = self._to_torch(batch.player_id)
        legal = self._to_torch(batch.legal)
        action_oh = np.eye(10)[batch.action]

        for player in range(_NUM_PLAYERS):
            reward = _rewards[:, :, player]  # [T, B, Player]
            v_target_, has_played, policy_target_ = v_trace(
                _v_target,
                batch.valid,
                batch.player_id,
                batch.policy,
                _pi,
                _log_policy_reg,
                _player_others(batch.player_id, batch.valid, player),
                action_oh,
                reward,
                player,
                lambda_=1.0,
                c=self.config.c_vtrace,
                rho=np.inf,
                eta=self.config.eta_reward_transform,
            )
            v_target_ = self._to_torch(np.array(v_target_))
            has_played = self._to_torch(np.array(has_played))
            policy_target_ = self._to_torch(np.array(policy_target_))

            v_target_list.append(v_target_)
            has_played_list.append(has_played)
            v_trace_policy_target_list.append(policy_target_)

        loss_v = get_loss_v([v] * _NUM_PLAYERS, v_target_list, has_played_list)

        is_vector = torch.unsqueeze(valid, axis=-1)
        importance_sampling_correction = [is_vector] * _NUM_PLAYERS
        # Uses v-trace to define q-values for Nerd
        loss_nerd = get_loss_nerd(
            [logit] * _NUM_PLAYERS,
            [pi] * _NUM_PLAYERS,
            v_trace_policy_target_list,
            valid,
            player_id,
            legal,
            importance_sampling_correction,
            clip=self.config.nerd.clip,
            threshold=self.config.nerd.beta,
        )
        loss = loss_v + loss_nerd
        return loss

    def update_parameters(self, batch: Batch, alpha: float, update_target_net: bool):
        """A jitted pure-functional part of the `step`."""

        loss_val = self.loss(batch, alpha)

        if loss_val.item() > 5:
            batch.save(f"debug/{self.learner_steps}-batch.bt")

        loss_val.backward()

        nn.utils.clip_grad.clip_grad_value_(
            self.params.parameters(), self.config.clip_gradient
        )

        # Update `params`` using the computed gradient.
        self.optimizer.step()

        # Update `params_target` towards `params`.
        self.optimizer_target.step()

        # Rolls forward the prev and prev_ params if update_target_net is 1.
        # pyformat: disable
        if update_target_net:
            print(f"Updating regularization nets @ {self.learner_steps:,}")
            self.params_prev_.load_state_dict(self.params_prev.state_dict())
            self.params_prev.load_state_dict(self.params_target.state_dict())

        self.params_actor.load_state_dict(self.params.state_dict())

        draws = abs(batch.rewards).sum(0).sum(1) == 0
        draw_ratio = draws.sum() / batch.valid.shape[1]

        logs = {
            "loss": loss_val,
            "draw_ratio": draw_ratio,
        }
        return logs
