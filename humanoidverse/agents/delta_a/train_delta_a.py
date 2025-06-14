import torch
import torch.nn as nn
import torch.optim as optim

from humanoidverse.agents.modules.ppo_modules import PPOActor, PPOCritic
from humanoidverse.agents.modules.data_utils import RolloutStorage
from humanoidverse.envs.base_task.base_task import BaseTask
from humanoidverse.agents.base_algo.base_algo import BaseAlgo
from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback
from humanoidverse.utils.average_meters import TensorAverageMeterDict

from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
import time
import os
import statistics
from collections import deque
from hydra.utils import instantiate
from loguru import logger
from rich.progress import track
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
console = Console()


from humanoidverse.agents.ppo.ppo import PPO
from humanoidverse.envs.base_task.base_task import BaseTask
from pathlib import Path
from omegaconf import OmegaConf
from humanoidverse.envs.motion_tracking.motion_tracking import LeggedRobotMotionTracking
from humanoidverse.utils.helpers import pre_process_config
from humanoidverse.agents.base_algo.base_algo import BaseAlgo 
from hydra.utils import instantiate

from humanoidverse.agents.delta_dynamics.delta_dynamics_model import DeltaDynamics_NN

class PPODeltaA(PPO):
    def __init__(self,
                 env: BaseTask,
                 config,
                 log_dir=None,
                 device='cpu'):
        super().__init__(env, config, log_dir, device)
        
        # Store the checkpoint paths for later loading
        self.policy_checkpoint_path = config.policy_checkpoint
        self.delta_checkpoint_path = config.delta_checkpoint

        # ----------------- UNCOMMENT THIS FOR ANALYTIC SEARCH FOR OPTIMAL ACTION BASED ON DELTA_A -----------------
        # if not hasattr(env, 'loaded_extra_policy'):
        #     setattr(env, 'loaded_extra_policy', self.loaded_policy)
        # if not hasattr(env.loaded_extra_policy, 'eval_policy'):
        #     setattr(env.loaded_extra_policy, 'eval_policy', self.loaded_policy._get_inference_policy())

        # ----------------- UNCOMMENT THIS FOR ANALYTIC SEARCH FOR OPTIMAL ACTION BASED ON DELTA_A -----------------    
        
    
    def setup(self):
        # Call parent setup first to create actor and critic
        super().setup()
        
        # Now load the pretrained weights
        self._load_pretrain_policy(self.policy_checkpoint_path)
        self.delta_model = self._load_delta_model(self.delta_checkpoint_path)

    # def _actor_act_step(self, obs_dict):
    #     actions = self.actor.act(obs_dict["actor_obs"])
    #     return self.actor.act_inference(obs_dict["actor_obs"])

    def _load_pretrain_policy(self,checkpoint_path):
        #Load pre-trained weights
        checkpoint=torch.load(checkpoint_path)
        self.actor.load_state_dict(checkpoint['actor_model_state_dict'])

        #Load critic weights
        self.critic.load_state_dict(checkpoint['critic_model_state_dict'])

        logger.info(f"Loaded pretrained policy from {checkpoint_path}")

    def _load_delta_model(self,checkpoint_path):
        #Create model
        self.input_dim = self.env.get_input_dim()
        self.output_dim = self.env.get_delta_output_dim()
        delta_model=DeltaDynamics_NN(self.input_dim,self.output_dim).to(self.device)

        #Loading weight
        checkpoint=torch.load(checkpoint_path)
        delta_model.load_state_dict(checkpoint['delta_dynamics'])

        #Freeze parameters
        for param in delta_model.parameters():
            param.requires_grad=False
        delta_model.eval()

        logger.info(f"Loaded delta policy from {checkpoint_path}")
        return delta_model


    def _rollout_step(self, obs_dict):
        # import ipdb; ipdb.set_trace()
        # self._eval_mode()
        # self.eval_policy = self._get_inference_policy()
        with torch.inference_mode():
            for i in range(self.num_steps_per_env):
                # Compute the actions and values
                # actions = self.actor.act(obs_dict["actor_obs"]).detach()

                policy_state_dict = {}
                policy_state_dict = self._actor_rollout_step(obs_dict, policy_state_dict)
                values = self._critic_eval_step(obs_dict).detach()
                policy_state_dict["values"] = values

                ## Append states to storage
                for obs_key in obs_dict.keys():
                    self.storage.update_key(obs_key, obs_dict[obs_key])

                for obs_ in policy_state_dict.keys():
                    self.storage.update_key(obs_, policy_state_dict[obs_])
                actions = policy_state_dict["actions"]
                actor_state = {}
                actor_state["actions"] = actions
                
                ################ inference the policy ################
                #delta action
                with torch.no_grad():
                    delta_action=self.delta_model(obs_dict['actor_obs'])
                    actor_state['actions_closed_loop']=delta_action

                

                # print('rollout_step actor_obs: ', obs_dict['actor_obs'])
                # print('rollout_step closed_loop_actor_obs: ', obs_dict['closed_loop_actor_obs'])
                # print('rollout_step actions: ', actions)
                # print('rollout_step closed_loop_actions: ', actor_state['actions_closed_loop'])

                ######################################################

                obs_dict, rewards, dones, infos = self.env.step(actor_state)
                # critic_obs = privileged_obs if privileged_obs is not None else obs
                for obs_key in obs_dict.keys():
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                rewards, dones = rewards.to(self.device), dones.to(self.device)

                self.episode_env_tensors.add(infos["to_log"])
                rewards_stored = rewards.clone().unsqueeze(1)
                if 'time_outs' in infos:
                    rewards_stored += self.gamma * policy_state_dict['values'] * infos['time_outs'].unsqueeze(1).to(self.device)
                assert len(rewards_stored.shape) == 2
                self.storage.update_key('rewards', rewards_stored)
                self.storage.update_key('dones', dones.unsqueeze(1))
                self.storage.increment_step()

                self._process_env_step(rewards, dones, infos)

                if self.log_dir is not None:
                    # Book keeping
                    if 'episode' in infos:
                        self.ep_infos.append(infos['episode'])
                    self.cur_reward_sum += rewards
                    self.cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    self.cur_reward_sum[new_ids] = 0
                    self.cur_episode_length[new_ids] = 0

            self.stop_time = time.time()
            self.collection_time = self.stop_time - self.start_time
            self.start_time = self.stop_time
            
            # prepare data for training

            returns, advantages = self._compute_returns(
                last_obs_dict=obs_dict,
                policy_state_dict=dict(values=self.storage.query_key('values'), 
                dones=self.storage.query_key('dones'), 
                rewards=self.storage.query_key('rewards'))
            )
            self.storage.batch_update_data('returns', returns)
            self.storage.batch_update_data('advantages', advantages)

        return obs_dict
    

    def _pre_eval_env_step(self, actor_state: dict):
        actions = self.eval_policy(actor_state["obs"]['actor_obs'])
        with torch.no_grad():
            actions_closed_loop = self.delta_model(actor_state['obs']['actor_obs'])

        actor_state.update({"actions": actions, "actions_closed_loop": actions_closed_loop})
        # actor_state.update({"actions": actions, "actions_closed_loop": actions_closed_loop, "current_closed_loop_actor_obs": actor_state['obs']['closed_loop_actor_obs']})
        # print("updated closed loop actor obs: ", actor_state['current_closed_loop_actor_obs'])
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state
