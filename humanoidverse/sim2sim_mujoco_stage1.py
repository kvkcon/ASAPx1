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
    model_xml_path = "/home/bbw/ASAPx1/humanoidverse/assets/robots/x1_29dof.xml"  # Adjust this path to the robot XML file
    output_path = "/home/bbw/ASAPx1/inference_results/boxlift_inference.pkl"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load ONNX model
    print(f"Loading ONNX model from {onnx_path}")
    ort_session = ort.InferenceSession(onnx_path)
    
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
    history_window = 3  # Assuming the policy uses some history (adjust as needed)
    
    # Storage for recorded states
    states = []
    
    # Prepare observation history (if needed by the policy)
    obs_history = []
    
    # Reset simulation
    mujoco.mj_resetData(model, data)
    
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
        
        # Prepare observation for policy (format according to your policy's requirements)
        # This is a simplified example - you'll need to adapt this to your specific observation format
        base_ang_vel = wtbase
        projected_gravity = np.array([0, 0, -9.81])  # Simple approximation, adjust as needed
        dof_pos = qtjoints
        dof_vel = qtvel
        actions = np.zeros_like(qtjoints)  # Initial action or previous action
        ref_motion_phase = np.array([step / total_steps])  # Simple phase approximation
        
        # Create observation dictionary (adjust according to your model's input format)
        obs_dict = {
            'base_ang_vel': base_ang_vel,
            'projected_gravity': projected_gravity,
            'dof_pos': dof_pos,
            'dof_vel': dof_vel,
            'actions': actions,
            'ref_motion_phase': ref_motion_phase,
        }
        
        # Add history if needed
        if len(obs_history) >= history_window:
            # Create history vector (format according to your policy requirements)
            # This is just a placeholder - adjust based on your actual history format
            history_actor = np.concatenate([h for h in obs_history[-history_window:]])
            obs_dict['history_actor'] = history_actor
        
        # Convert observation dict to network input format
        # This is a simplified example - adjust according to your policy's input requirements
        network_input = np.concatenate([
            obs_dict['base_ang_vel'],
            obs_dict['projected_gravity'],
            obs_dict['dof_pos'],
            obs_dict['dof_vel'],
            obs_dict['actions'],
            obs_dict['ref_motion_phase'],
        ])
        
        if 'history_actor' in obs_dict:
            network_input = np.concatenate([network_input, obs_dict['history_actor']])
        
        # Get action from policy
        input_name = ort_session.get_inputs()[0].name
        action = ort_session.run(None, {input_name: network_input.reshape(1, -1).astype(np.float32)})[0][0]
        
        # Update observation history
        obs_history.append(np.concatenate([
            base_ang_vel, projected_gravity, dof_pos, dof_vel, action, ref_motion_phase
        ]))
        if len(obs_history) > history_window:
            obs_history.pop(0)
        
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
    
    # Convert joint positions to pose_aa format (adjust as needed)
    # This is a placeholder - you may need to convert your joint angles to axis-angle format
    pose_aa = np.zeros((num_frames, 31, 3))  # Placeholder
    
    dof = np.array([s['qt'] for s in states])
    root_rot = np.array([s['qtbase'] for s in states])
    
    # Convert to SMPL joints (placeholder - adjust as needed)
    smpl_joints = np.zeros((num_frames, 24, 3))  # Placeholder
    
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