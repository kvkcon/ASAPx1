#!/bin/bash

# Directory containing motion files
MOTION_DIR="humanoidverse/data/motions/x1_29dof/Test-amass-dance/asset1"

# Base command for training
BASE_CMD="python humanoidverse/train_agent.py \
  +simulator=isaacgym \
  +exp=motion_tracking \
  +domain_rand=NO_domain_rand \
  +rewards=motion_tracking/reward_motion_tracking_dm_2real \
  +robot=x1/x1_29dof \
  +terrain=terrain_locomotion_plane \
  +obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history \
  num_envs=4096 \
  project_name=MotionTracking \
  rewards.reward_penalty_curriculum=True \
  rewards.reward_penalty_degree=0.00001 \
  env.config.resample_motion_when_training=False \
  env.config.termination.terminate_when_motion_far=True \
  env.config.termination_curriculum.terminate_when_motion_far_curriculum=True \
  env.config.termination_curriculum.terminate_when_motion_far_threshold_min=0.3 \
  env.config.termination_curriculum.terminate_when_motion_far_curriculum_degree=0.000025 \
  robot.asset.self_collisions=0"

# Loop through each motion file in the directory
for MOTION_FILE in "$MOTION_DIR"/*.pkl; do
    # Extract just the filename without extension
    FILENAME=$(basename "$MOTION_FILE" .pkl)
    
    echo "Starting training for motion: $FILENAME"
    
    # Run the training command with the current motion file
    # Training for each motion will save weights every 2000 steps and stop at 10000 steps
    nohup $BASE_CMD \
      experiment_name="${FILENAME}_human" \
      robot.motion.motion_file="$MOTION_FILE" \
      max_iterations=10000 > "nohup_${FILENAME}.out" 2>&1
    
    echo "Completed training for motion: $FILENAME"
    
    # Optional: add a short pause between training runs
    sleep 5
done

echo "All motion training complete!"