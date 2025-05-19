#!/bin/bash

# Directory containing motion files
MOTION_DIR="humanoidverse/data/motions/x1_29dof/Test-amass-dance/asset1"

# Loop through each motion file in the directory
for MOTION_FILE in "$MOTION_DIR"/*.pkl; do
    # Extract just the filename without extension
    FILENAME=$(basename "$MOTION_FILE" .pkl)
    
    echo "Starting training for motion: $FILENAME"
    
    # Start the training process in the background
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
    
    # Get the process ID of the training
    TRAIN_PID=$!
    
    echo "Training process started with PID: $TRAIN_PID for motion: $FILENAME"
    
    # Monitor the iterations by checking the log file
    ITERATION=0
    MAX_ITERATION=10000
    CHECK_INTERVAL=30  # Check every 30 seconds
    
    while [ $ITERATION -lt $MAX_ITERATION ]; do
        # Check if the process is still running
        if ! ps -p $TRAIN_PID > /dev/null; then
            echo "Training process for $FILENAME ended unexpectedly"
            break
        fi
        
        # Get the current iteration from the log file
        # This pattern will need to be adjusted based on your actual log output format
        if [ -f "training_${FILENAME}.log" ]; then
            # Look for lines containing iteration information
            CURRENT_ITER=$(grep -o "Iteration: [0-9]\+" "training_${FILENAME}.log" | tail -1 | awk '{print $2}')
            
            # If we found an iteration number, update our counter
            if [ ! -z "$CURRENT_ITER" ]; then
                ITERATION=$CURRENT_ITER
                echo "Current iteration for $FILENAME: $ITERATION"
            fi
        fi
        
        # If we've reached our target iteration, kill the process
        if [ $ITERATION -ge $MAX_ITERATION ]; then
            echo "Reached target iteration of $MAX_ITERATION for $FILENAME. Stopping training."
            kill $TRAIN_PID
            break
        fi
        
        # Wait before checking again
        sleep $CHECK_INTERVAL
    done
    
    echo "Completed training for motion: $FILENAME"
    
    # Optional: add a short pause between training runs
    sleep 5
done

echo "All motion training complete!"