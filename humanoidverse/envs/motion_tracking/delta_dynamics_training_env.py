import torch
import numpy as np
from pathlib import Path
import os
from humanoidverse.envs.motion_tracking.motion_tracking import LeggedRobotMotionTracking
from isaac_utils.rotations import (
    my_quat_rotate,
    calc_heading_quat_inv,
    calc_heading_quat,
    quat_mul,
    quat_conjugate,
    quat_to_angle_axis,
    quat_rotate_inverse,
    xyzw_to_wxyz,
    wxyz_to_xyzw
)
from termcolor import colored
from loguru import logger
from scipy.spatial.transform import Rotation as sRot
import joblib

class LeggedRobotDeltaDynamics(LeggedRobotMotionTracking):
    def __init__(self, config, device):
        self.init_done = False
        super().__init__(config, device)
        self._init_delta_dynamics_config()
        self.init_done = True
        
    def _init_delta_dynamics_config(self):
        """Initialize delta dynamics specific configurations"""
        # Delta dynamics specific parameters
        self.delta_state_buffer = {}
        self.target_state_buffer = {}
        self.motion_data_cache = {}
        
        # Initialize state dimensions for delta computation
        self.state_dims = {
            'motion_dof_pos': self.config.robot.dof_obs_size,
            'motion_dof_vel': self.config.robot.dof_obs_size,
            'motion_base_pos_xyz': 3,
            'motion_base_lin_vel': 3,
            'motion_base_ang_vel': 3,
            'motion_base_quat': 4,
        }
        
        logger.info("Delta dynamics environment initialized")

    def get_input_dim(self):
        """Get input dimension for delta dynamics network"""
        obs_key = 'delta_dynamics_input_obs'
        if obs_key in self.config.obs.obs_dict:
            print(f"delta_dynamics_input_obs dim: {self.config.obs.obs_dict[obs_key]['dim']}")
            return self.config.obs.obs_dict[obs_key]['dim']
        else:
            # Define components for delta dynamics input
            components = [
                'base_ang_vel',
                'projected_gravity', 
                'dof_pos',
                'dof_vel',
                'actions',
                'ref_motion_phase'
            ]
            obs_dims_dict = {key: value for key, value in self.config.obs.obs_dims.items()}
            print("obs_dims_dict", obs_dims_dict)
            dim = sum(obs_dims_dict.get(key, 0) for key in components)
            print("dim1", dim)
            
            # Add history component if configured
            if 'history_actor' in self.config.obs.obs_auxiliary:
                history_config = self.config.obs.obs_auxiliary['history_actor']
                history_dim = sum(obs_dims_dict.get(key, 0) * length for key, length in history_config.items())
                dim += history_dim
            print("dim2", dim)
            return dim

    def get_output_dim(self):
        """Get output dimension for delta dynamics network"""
        return sum(self.state_dims.values())
    
    def parse_delta(self, delta, prefix='pred'):
        """Parse delta tensor into state components"""
        state_idx = 0
        delta_state = {}
        for key, dim in self.state_dims.items():
            delta_state[key.replace('motion_', f'{prefix}_')] = delta[:, state_idx:state_idx+dim]
            state_idx += dim
        return delta_state

    def update_delta(self, delta_state_items):
        """Update state with delta predictions"""
        pred_state = {
            'dof_pos': self.simulator.dof_pos + delta_state_items['pred_dof_pos'],
            'dof_vel': self.simulator.dof_vel + delta_state_items['pred_dof_vel'],
            'base_pos_xyz': self.simulator.robot_root_states[:, 0:3] + delta_state_items['pred_base_pos_xyz'],
            'base_lin_vel': self.simulator.robot_root_states[:, 7:10] + delta_state_items['pred_base_lin_vel'],
            'base_ang_vel': self.simulator.robot_root_states[:, 10:13] + delta_state_items['pred_base_ang_vel'],
            'base_quat': self.simulator.robot_root_states[:, 3:7] + delta_state_items['pred_base_quat'],
        }
        # Normalize quaternion
        pred_state['base_quat'] = pred_state['base_quat'] / torch.norm(pred_state['base_quat'], dim=-1, keepdim=True)
        return pred_state

    def assemble_delta(self, state_dict):
        """Assemble state dictionary into delta tensor for loss computation"""
        delta_components = []
        for key in self.state_dims.keys():
            state_key = key.replace('motion_', '')
            if state_key in state_dict:
                delta_components.append(state_dict[state_key])
            else:
                # Handle missing keys with zeros
                delta_components.append(torch.zeros(self.num_envs, self.state_dims[key], device=self.device))
        
        return torch.cat(delta_components, dim=-1)

    def get_target_state_from_motion(self, motion_times=None):
        """Get target state from motion library for delta dynamics training"""
        if motion_times is None:
            motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times
        
        offset = self.env_origins
        motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)
        
        target_state = {
            'motion_dof_pos': motion_res['dof_pos'],
            'motion_dof_vel': motion_res['dof_vel'],
            'motion_base_pos_xyz': motion_res['root_pos'],
            'motion_base_lin_vel': motion_res['root_vel'],
            'motion_base_ang_vel': motion_res['root_ang_vel'],
            'motion_base_quat': motion_res['root_rot'],
        }
        
        return target_state

    def compute_delta_loss_components(self, pred_state, target_state):
        """Compute individual loss components for delta dynamics"""
        loss_components = {}
        
        # DOF position loss
        loss_components['loss_dof_pos'] = torch.nn.functional.mse_loss(
            pred_state['dof_pos'], target_state['motion_dof_pos']
        )
        
        # DOF velocity loss
        loss_components['loss_dof_vel'] = torch.nn.functional.mse_loss(
            pred_state['dof_vel'], target_state['motion_dof_vel']
        )
        
        # Base linear velocity loss
        loss_components['loss_base_lin_vel'] = torch.nn.functional.mse_loss(
            pred_state['base_lin_vel'], target_state['motion_base_lin_vel']
        )
        
        # Base angular velocity loss
        loss_components['loss_base_ang_vel'] = torch.nn.functional.mse_loss(
            pred_state['base_ang_vel'], target_state['motion_base_ang_vel']
        )
        
        # Base position loss
        loss_components['loss_base_pos'] = torch.nn.functional.mse_loss(
            pred_state['base_pos_xyz'], target_state['motion_base_pos_xyz']
        )
        
        # Base quaternion loss (special handling for quaternions)
        loss_components['loss_base_quat'] = torch.nn.functional.mse_loss(
            pred_state['base_quat'], target_state['motion_base_quat']
        )
        
        # Total loss
        loss_components['total_loss'] = sum(loss_components.values())
        
        return loss_components

    def step_with_delta(self, actor_state):
        """Environment step function with delta dynamics integration"""
        # Extract delta dynamics prediction if available
        if 'delta_state_items' in actor_state:
            # Apply delta state update
            pred_state = self.update_delta(actor_state['delta_state_items'])
            
            # Update simulator state with predictions
            self._apply_predicted_state(pred_state)
        
        # Call parent step function
        return super().step(actor_state)
    
    def _apply_predicted_state(self, pred_state):
        """Apply predicted state to simulator"""
        # Update DOF positions and velocities
        self.simulator.dof_pos[:] = pred_state['dof_pos']
        self.simulator.dof_vel[:] = pred_state['dof_vel']
        
        # Update root states
        self.simulator.robot_root_states[:, 0:3] = pred_state['base_pos_xyz']
        self.simulator.robot_root_states[:, 3:7] = pred_state['base_quat']
        self.simulator.robot_root_states[:, 7:10] = pred_state['base_lin_vel']
        self.simulator.robot_root_states[:, 10:13] = pred_state['base_ang_vel']

    def _compute_observations(self):
        """Override to include delta dynamics specific observations"""
        super()._compute_observations()
        
        # Add delta dynamics input observation
        if 'delta_dynamics_input_obs' in self.config.obs.obs_dict:
            self._compute_delta_dynamics_input_obs()

    def _compute_delta_dynamics_input_obs(self):
        """Compute observation for delta dynamics network input"""
        # Combine relevant observations for delta dynamics
        obs_components = []
        
        # Add base angular velocity
        if 'base_ang_vel' in self.obs_buf_dict:
            obs_components.append(self.obs_buf_dict['base_ang_vel'])
        
        # Add projected gravity
        if 'projected_gravity' in self.obs_buf_dict:
            obs_components.append(self.obs_buf_dict['projected_gravity'])
            
        # Add DOF positions and velocities
        if 'dof_pos' in self.obs_buf_dict:
            obs_components.append(self.obs_buf_dict['dof_pos'])
        if 'dof_vel' in self.obs_buf_dict:
            obs_components.append(self.obs_buf_dict['dof_vel'])
            
        # Add actions
        if 'actions' in self.obs_buf_dict:
            obs_components.append(self.obs_buf_dict['actions'])
            
        # Add reference motion phase
        if hasattr(self, '_ref_motion_phase'):
            obs_components.append(self._ref_motion_phase)
        
        # Concatenate all components
        if obs_components:
            self.obs_buf_dict['delta_dynamics_input_obs'] = torch.cat(obs_components, dim=-1)
        else:
            # Fallback to zeros if no components available
            self.obs_buf_dict['delta_dynamics_input_obs'] = torch.zeros(
                self.num_envs, self.get_input_dim(), device=self.device
            )

    def load_motion_data_for_training(self, motion_file):
        """Load motion data for supervised learning"""
        if motion_file not in self.motion_data_cache:
            motion_path = os.path.join(os.path.dirname(__file__), '..', motion_file)
            if os.path.exists(motion_path):
                motion_data = joblib.load(motion_path)
                self.motion_data_cache[motion_file] = motion_data
                logger.info(f"Loaded motion data from {motion_path}")
            else:
                logger.error(f"Motion file not found: {motion_path}")
                return None
        
        return self.motion_data_cache[motion_file]

    def get_training_batch(self, motion_file, batch_size=None):
        """Get a batch of training data for delta dynamics"""
        motion_data = self.load_motion_data_for_training(motion_file)
        if motion_data is None:
            return None
            
        if batch_size is None:
            batch_size = self.num_envs
            
        # Extract frames data
        frames = motion_data.get('frames', [])
        if len(frames) == 0:
            logger.warning("No frames found in motion data")
            return None
            
        # Sample random frames for training
        num_frames = len(frames)
        indices = torch.randint(0, num_frames, (batch_size,))
        
        batch_data = []
        for idx in indices:
            frame_data = frames[idx.item()]
            batch_data.append(frame_data)
            
        return batch_data

    def reset_for_delta_training(self):
        """Reset environment specifically for delta dynamics training"""
        # Reset to random motion states
        self._resample_motion_times(torch.arange(self.num_envs, device=self.device))
        
        # Reset environment
        obs_dict = self.reset_all()
        
        # Get target state from motion
        target_state = self.get_target_state_from_motion()
        
        return obs_dict, target_state

    def _post_physics_step(self):
        """Override post physics step for delta dynamics specific functionality"""
        super()._post_physics_step()
        
        # Store current state for delta computation if needed
        if hasattr(self, 'store_delta_states') and self.store_delta_states:
            self._store_current_state_for_delta()
    
    def _store_current_state_for_delta(self):
        """Store current state for delta dynamics computation"""
        self.delta_state_buffer = {
            'dof_pos': self.simulator.dof_pos.clone(),
            'dof_vel': self.simulator.dof_vel.clone(),
            'base_pos_xyz': self.simulator.robot_root_states[:, 0:3].clone(),
            'base_lin_vel': self.simulator.robot_root_states[:, 7:10].clone(),
            'base_ang_vel': self.simulator.robot_root_states[:, 10:13].clone(),
            'base_quat': self.simulator.robot_root_states[:, 3:7].clone(),
        }