import os
import numpy as np
import pickle
import mujoco
import onnxruntime as ort
import time
from pathlib import Path

def main():
    # Path configurations
    onnx_path = "/home/bbw/ASAPx1/logs/MotionTracking/20250509_232245-MotionTracking_Boxlift_29dof_alphabet_changed_urdf_poseReward_reduce_correctHeadlink-motion_tracking-x1/exported/model_14000.onnx"
    model_xml_path = "/home/bbw/ASAPx1/humanoidverse/data/robots/x1/x1.xml"
    output_path = "/home/bbw/ASAPx1/inference_results/boxlift_inference.pkl"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load ONNX model
    print(f"Loading ONNX model from {onnx_path}")
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    print(f"Model input name: {input_name}")
    
    # Load MuJoCo model
    print(f"Loading MuJoCo model from {model_xml_path}")
    model = mujoco.MjModel.from_xml_path(model_xml_path)
    data = mujoco.MjData(model)
    
    # Initialize simulation parameters
    sim_duration = 10.0  # seconds
    control_timestep = 1/30.0  # 30 Hz control frequency
    physics_timestep = model.opt.timestep
    steps_per_control = int(control_timestep / physics_timestep)
    
    total_steps = int(sim_duration / control_timestep)
    
    # Observation dimensions based on logs
    obs_dims = {
        'base_lin_vel': 3, 
        'base_ang_vel': 3, 
        'projected_gravity': 3, 
        'dof_pos': 29, 
        'dof_vel': 29, 
        'actions': 29, 
        'dif_local_rigid_body_pos': 93, 
        'local_ref_rigid_body_pos': 93, 
        'ref_motion_phase': 1
    }
    
    # Actor observation includes these components:
    actor_obs_keys = [
        'base_ang_vel',      # 3
        'projected_gravity', # 3
        'dof_pos',           # 29
        'dof_vel',           # 29
        'actions',           # 29
        'ref_motion_phase'   # 1
    ]
    
    # Calculate the size of a single actor_obs without history
    single_actor_obs_size = sum(obs_dims[key] for key in actor_obs_keys)  # Should be 94
    
    # Based on the logs and errors, we need 5 total observations (current + 4 history frames)
    # to reach the expected 470 dimensions (5 * 94 = 470)
    history_window = 5
    
    # Initialize history
    actor_obs_history = []
    
    # Storage for recorded states
    states = []
    
    # Reset simulation
    mujoco.mj_resetData(model, data)
    
    # Initial action
    actions = np.zeros(obs_dims['actions'])
    
    print(f"Starting simulation for {sim_duration} seconds ({total_steps} steps)")
    
    for step in range(total_steps):
        # Get current state
        ptbase = data.qpos[:3].copy()  # Base position
        qtbase = data.qpos[3:7].copy()  # Base orientation (quaternion)
        qtjoints = data.qpos[7:].copy()  # Joint positions
        
        vtbase = data.qvel[:3].copy()  # Base linear velocity
        wtbase = data.qvel[3:6].copy()  # Base angular velocity
        qtvel = data.qvel[6:].copy()  # Joint velocities
        
        # Record state
        current_state = {
            'ptbase': ptbase,
            'vtbase': vtbase,
            'qtbase': qtbase,  # αtbase (quaternion)
            'wtbase': wtbase,  # ωtbase
            'qt': qtjoints,    # qt (joint positions)
            'qt_vel': qtvel    # q̇t (joint velocities)
        }
        states.append(current_state)

        gravity_vec = np.array([0, 0, -9.81], dtype=np.float64)
        # # 注意：MuJoCo四元数格式是[x,y,z,w]，而PyTorch格式可能是[w,x,y,z]，请确认格式
        # # 如果需要转换: qtbase_wxyz = np.array([qtbase[3], qtbase[0], qtbase[1], qtbase[2]])
        projected_gravity = mujoco_quat_rotate_inverse(qtbase, gravity_vec)
        
        # Prepare observation for policy
        obs_dict = {
            'base_lin_vel': vtbase,  # Not used in actor_obs but recorded
            'base_ang_vel': wtbase,
            'projected_gravity': np.array([0, 0, -9.81]),  # Simple approximation, adjust if needed
            'dof_pos': qtjoints,
            'dof_vel': qtvel,
            'actions': actions,
            'ref_motion_phase': np.array([step / total_steps])  # Simple phase approximation # but ref_motion_phase in motion_tracking is motion_times / self._ref_motion_length
        }
        
        # Create the current actor observation (without history)
        current_actor_obs = np.concatenate([obs_dict[key] for key in actor_obs_keys])
        
        # Add to history
        actor_obs_history.append(current_actor_obs)
        if len(actor_obs_history) > history_window:
            actor_obs_history.pop(0)
        
        # If we don't have enough history yet, duplicate the current observation
        while len(actor_obs_history) < history_window:
            actor_obs_history.append(current_actor_obs)
        
        # Construct the full actor_obs with history (all observations, not just history)
        # It appears the format might be [current_obs, history_frame_1, history_frame_2, ...]
        full_actor_obs = np.concatenate(actor_obs_history)
        
        # Check the shape before running inference
        if full_actor_obs.shape[0] != 470:
            print(f"Warning: actor_obs shape is {full_actor_obs.shape[0]}, expected 470")
            print(f"Current obs size: {current_actor_obs.shape[0]}, History size: {full_actor_obs.shape[0] - current_actor_obs.shape[0]}")
            # Try to adjust if there's a mismatch
            if full_actor_obs.shape[0] < 470:
                # Pad with zeros if too small
                pad_size = 470 - full_actor_obs.shape[0]
                full_actor_obs = np.pad(full_actor_obs, (0, pad_size), 'constant')
            elif full_actor_obs.shape[0] > 470:
                # Truncate if too large
                full_actor_obs = full_actor_obs[:470]
        
        # Run inference
        try:
            action = ort_session.run(None, {input_name: full_actor_obs.reshape(1, -1).astype(np.float32)})[0][0]
            # Store the action for the next step's observation
            actions = action.copy()
        except Exception as e:
            print(f"Error during inference: {e}")
            print(f"Input shape: {full_actor_obs.shape}")
            break
        
        # Apply action to control
        data.ctrl[:] = action
        
        # Step simulation
        for _ in range(steps_per_control):
            mujoco.mj_step(model, data)
        
        if step % 30 == 0:  # Print progress every second
            print(f"Simulation progress: {step}/{total_steps} steps ({step/total_steps*100:.1f}%)")
    
    print("Simulation completed")
    
    # Prepare the final data structure
    motion_name = "boxlift_inference"
    num_frames = len(states)
    
    # Extract data arrays
    root_trans = np.array([s['ptbase'] for s in states])
    
    # For pose_aa, we'd need to convert joint positions to axis-angle format
    # This is a placeholder, and you may need to implement the proper conversion
    pose_aa = np.zeros((num_frames, 31, 3))
    
    dof = np.array([s['qt'] for s in states])
    root_rot = np.array([s['qtbase'] for s in states])
    
    # For SMPL joints, you'd need the proper forward kinematics
    # This is a placeholder
    smpl_joints = np.zeros((num_frames, 24, 3))
    
    # Create the final dictionary structure
    final_data = {
        motion_name: {
            'root_trans_offset': root_trans.astype(np.float64),
            'pose_aa': pose_aa.astype(np.float32),
            'dof': dof.astype(np.float32),
            'root_rot': root_rot.astype(np.float64),
            'smpl_joints': smpl_joints.astype(np.float32),
            'fps': 30
        }
    }
    
    # Save to pkl file
    print(f"Saving results to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(final_data, f)
    
    print("Done!")

# 在mujoco代码中实现quat_rotate_inverse的等效功能
def mujoco_quat_rotate_inverse(quat, vec):
    """
    将向量从世界坐标系旋转到以四元数定义的局部坐标系
    
    Args:
        quat: 四元数 [x, y, z, w] 格式
        vec: 向量 [x, y, z]
        
    Returns:
        旋转后的向量 [x, y, z]
    """
    # 确保使用float32
    quat = np.array(quat, dtype=np.float32)
    vec = np.array(vec, dtype=np.float32)
    
    # 提取四元数分量
    q_x, q_y, q_z, q_w = quat
    
    # 计算旋转
    a = vec * (2.0 * q_w**2 - 1.0)
    b = np.cross(quat[:3], vec) * (q_w * 2.0)
    c = quat[:3] * (np.dot(quat[:3], vec) * 2.0)
    
    return a - b + c

if __name__ == "__main__":
    main()