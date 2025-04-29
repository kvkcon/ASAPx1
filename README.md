<h1 align="center"> ASAP: Aligning Simulation and Real-World Physics for 

Learning Agile Humanoid Whole-Body Skills </h1>


[![IsaacGym](https://img.shields.io/badge/IsaacGym-Preview4-b.svg)](https://developer.nvidia.com/isaac-gym) [![IsaacSim](https://img.shields.io/badge/IsaacSim-4.2.0-b.svg)](https://docs.isaacsim.omniverse.nvidia.com/4.2.0/index.html) [![IsaacSim](https://img.shields.io/badge/Genesis-0.2.1-b.svg)](https://docs.isaacsim.omniverse.nvidia.com/4.2.0/index.html) [![Linux platform](https://img.shields.io/badge/Platform-linux--64-orange.svg)](https://ubuntu.com/blog/tag/22-04-lts) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()


<img src="https://agile.human2humanoid.com/static/images/asap-preview-gif-480P.gif" width="400px"/>
<video width="400px" controls>
  <source src="asset/walk_with_arm.mp4" type="video/mp4">
</video>

</div>

## Official TODO
- [x] Release code backbone
- [x] Release phase-based motion tracking training pipeline
- [ ] Release ASAP motion datasets
- [ ] Release motion retargeting pipeline
- [ ] Release sim2sim in MuJoCo
- [ ] Release sim2real with UnitreeSDK
- [ ] Release ASAP delta action model training pipeline


# Installation

ASAP codebase is built on top of [HumanoidVerse](https://github.com/LeCAR-Lab/HumanoidVerse) (a multi-simulator framework for humanoid learning) and [Human2Humanoid](https://github.com/LeCAR-Lab/human2humanoid) (our prior work on humanoid whole-body tracking).

[HumanoidVerse](https://github.com/LeCAR-Lab/HumanoidVerse) allows you to train humanoid skills in multiple simulators, including IsaacGym, IsaacSim, and Genesis. Its key design logic is the separation and modularization of simulators, tasks, and algorithms, which enables smooth transfers between different simulators and the real world with minimum effort (just one line of code change). We leverage this framework to develop [ASAP](https://agile.human2humanoid.com/) and study how to best transfer policies across simulators and the real world.

See `README_origin.md`


# Motion Tracking Training on x1

Train a phase-based motion tracking policy to imitate dance motion from AMASS dataset

```bash
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
experiment_name=MotionTracking_Punchboxing \
robot.motion.motion_file="humanoidverse/data/motions/x1_29dof/Test-amass-dance/singles/0-Transitions_mocap_mazen_c3d_punchboxing_walk_poses.pkl" \
rewards.reward_penalty_curriculum=True \
rewards.reward_penalty_degree=0.00001 \
env.config.resample_motion_when_training=False \
env.config.termination.terminate_when_motion_far=True \
env.config.termination_curriculum.terminate_when_motion_far_curriculum=True \
env.config.termination_curriculum.terminate_when_motion_far_threshold_min=0.3 \
env.config.termination_curriculum.terminate_when_motion_far_curriculum_degree=0.000025 \
robot.asset.self_collisions=0
```

```bash
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py +simulator=isaacgym +exp=motion_tracking +domain_rand=NO_domain_rand +rewards=motion_tracking/reward_motion_tracking_dm_2real +robot=x1/x1_29dof +terrain=terrain_locomotion_plane +obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history num_envs=4096 project_name=MotionTracking experiment_name=MotionTracking_Punchboxing robot.motion.motion_file="humanoidverse/data/motions/x1_29dof/Test-amass-dance/0-Transitions_mocap_mazen_c3d_punchboxing_walk_poses.pkl" rewards.reward_penalty_curriculum=True rewards.reward_penalty_degree=0.00001 env.config.resample_motion_when_training=False env.config.termination.terminate_when_motion_far=True env.config.termination_curriculum.terminate_when_motion_far_curriculum=True env.config.termination_curriculum.terminate_when_motion_far_threshold_min=0.3 env.config.termination_curriculum.terminate_when_motion_far_curriculum_degree=0.000025 robot.asset.self_collisions=0
```

continue training from ckpt
```bash
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py +simulator=isaacgym +exp=motion_tracking +domain_rand=NO_domain_rand +rewards=motion_tracking/reward_motion_tracking_dm_2real +robot=x1/x1_29dof +terrain=terrain_locomotion_plane +obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history +checkpoint=logs/MotionTracking/20250422_114106-MotionTracking_Punchboxing_Leftleg-motion_tracking-x1/model_4000.pt  num_envs=4096 project_name=MotionTracking experiment_name=MotionTracking_Punchboxing_Leftleg robot.motion.motion_file="humanoidverse/data/motions/x1_29dof/Test-amass-dance/0-Transitions_mocap_mazen_c3d_punchboxing_walk_poses.pkl" rewards.reward_penalty_curriculum=True rewards.reward_penalty_degree=0.00001 env.config.resample_motion_when_training=False env.config.termination.terminate_when_motion_far=True env.config.termination_curriculum.terminate_when_motion_far_curriculum=True env.config.termination_curriculum.terminate_when_motion_far_threshold_min=0.3 env.config.termination_curriculum.terminate_when_motion_far_curriculum_degree=0.000025 robot.asset.self_collisions=0
```

After training, you can visualize the policy by:
```bash
python humanoidverse/eval_agent.py \
+checkpoint=logs/MotionTracking/xxxxxxxx_xxxxxxx-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/model_5800.pt
```

This is the visualization of the policy after traning 5800 iters. The policy is able to imitate the motion of Cristiano Ronaldo's Siuuu move. With more training, the policy will be more accurate and smooth (see the video in the [paper](https://arxiv.org/pdf/2502.01143)).

<img src="imgs/motion_tracking_5800.gif" width="400px"/>

# Citation
If you find our work useful, please consider citing us!

```bibtex
@article{he2025asap,
  title={ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills},
  author={He, Tairan and Gao, Jiawei and Xiao, Wenli and Zhang, Yuanhang and Wang, Zi and Wang, Jiashun and Luo, Zhengyi and He, Guanqi and Sobanbabu, Nikhil and Pan, Chaoyi and Yi, Zeji and Qu, Guannan and Kitani, Kris and Hodgins, Jessica and Fan, Linxi "Jim" and Zhu, Yuke and Liu, Changliu and Shi, Guanya},
  journal={arXiv preprint arXiv:2502.01143},
  year={2025}
}
```

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

