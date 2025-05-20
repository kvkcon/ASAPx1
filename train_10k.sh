#!/bin/bash

# Directory containing motion files
MOTION_DIR="humanoidverse/data/motions/x1_29dof/Test-amass-dance/asset2"

# Loop through each motion file in the directory
for MOTION_FILE in "$MOTION_DIR"/*.pkl; do
    # Extract just the filename without extension
    FILENAME=$(basename "$MOTION_FILE" .pkl)
    
    echo "Starting training for motion: $FILENAME"
    
    # Get current timestamp for log directory identification
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_DIR_PREFIX="logs/MotionTracking/${TIMESTAMP}-${FILENAME}_human-motion_tracking-x1"
    
    # Start the training process
    python humanoidverse/train_agent.py \
      +simulator=isaacgym \
      +exp=motion_tracking \
      +domain_rand=NO_domain_rand \
      +rewards=motion_tracking/reward_motion_tracking_dm_2real \
      +robot=x1/x1_29dof \
      +terrain=terrain_locomotion_plane \
      +obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history \
      num_envs=4096 \
      project_name=MotionTracking \
      experiment_name="${FILENAME}_human" \
      robot.motion.motion_file="$MOTION_FILE" \
      rewards.reward_penalty_curriculum=True \
      rewards.reward_penalty_degree=0.00001 \
      env.config.resample_motion_when_training=False \
      env.config.termination.terminate_when_motion_far=True \
      env.config.termination_curriculum.terminate_when_motion_far_curriculum=True \
      env.config.termination_curriculum.terminate_when_motion_far_threshold_min=0.3 \
      env.config.termination_curriculum.terminate_when_motion_far_curriculum_degree=0.000025 \
      robot.asset.self_collisions=0 > "training_${FILENAME}.log" 2>&1 &
    
    # Get the process ID
    TRAIN_PID=$!
    echo "Training process started with PID: $TRAIN_PID for motion: $FILENAME"
    
    # Wait for log directory to be created
    sleep 10
    
    # Find the actual log directory that was created
    LOG_DIR=$(find logs/MotionTracking -type d -name "*-${FILENAME}_human-motion_tracking-x1" | sort -r | head -n 1)
    echo "Log directory: $LOG_DIR"
    
    # Wait until we reach 10000 steps (5 checkpoints at 2000 steps each)
    TARGET_CHECKPOINT=5  # model_4.pt will be the 5th checkpoint (at 10000 steps)
    
    while true; do
        # Check if the process is still running
        if ! ps -p $TRAIN_PID > /dev/null; then
            echo "Training process for $FILENAME ended unexpectedly"
            break
        fi
        
        # Check if the target checkpoint exists
        if [ -f "$LOG_DIR/model_10000.pt" ]; then
            echo "Reached target checkpoint for $FILENAME (10000 steps). Stopping training."
            kill $TRAIN_PID
            sleep 2
            # Make sure the process is killed
            if ps -p $TRAIN_PID > /dev/null; then
                kill -9 $TRAIN_PID
            fi
            break
        fi
        
        # echo "Waiting for training to reach 10000 steps for $FILENAME..."
        sleep 1200
    done
    
    echo "Completed training for motion: $FILENAME"
    
    # Optional: add a short pause between training runs
    sleep 5
done

echo "All motion training complete!"