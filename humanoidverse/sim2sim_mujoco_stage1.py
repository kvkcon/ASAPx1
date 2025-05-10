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
    auxiliary_obs_dims = {
        'history_actor': 376, 
        'history_critic': 388
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
    single_actor_obs_size = sum(obs_dims[key] for key in actor_obs_keys)
    
    # The history window needed for actor_obs
    # The actor_obs is 470 in total, and single_actor_obs is (3+3+29+29+29+1) = 94
    # So we have 470 - 94 = 376 elements for history, which matches auxiliary_obs_dims['history_actor']
    # Since each historical observation is 94 elements, we need 376/94 ~ 4 history time steps
    history_window = 4
    
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
        
        # Prepare observation for policy
        obs_dict = {
            'base_lin_vel': vtbase,  # Not used in actor_obs but recorded
            'base_ang_vel': wtbase,
            'projected_gravity': np.array([0, 0, -9.81]),  # Simple approximation, adjust if needed
            'dof_pos': qtjoints,
            'dof_vel': qtvel,
            'actions': actions,
            'ref_motion_phase': np.array([step / total_steps])  # Simple phase approximation
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
        
        # Construct the full actor_obs with history
        history_flat = np.concatenate(actor_obs_history[:-1])  # All but the current observation
        full_actor_obs = np.concatenate([current_actor_obs, history_flat])
        
        # Check the shape before running inference
        if full_actor_obs.shape[0] != 470:
            print(f"Warning: actor_obs shape is {full_actor_obs.shape[0]}, expected 470")
            print(f"Current obs size: {current_actor_obs.shape[0]}, History size: {history_flat.shape[0]}")
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
            'root_trans_offset': root_trans,
            'pose_aa': pose_aa,
            'dof': dof,
            'root_rot': root_rot,
            'smpl_joints': smpl_joints,
            'fps': 30
        }
    }
    
    # Save to pkl file
    print(f"Saving results to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(final_data, f)
    
    print("Done!")

if __name__ == "__main__":
    main()