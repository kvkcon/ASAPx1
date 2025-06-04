#!/bin/bash

source /opt/conda/etc/profile.d/conda.sh
conda activate tram_env
cd ..
# 确保脚本接收到参数
if [ -z "$1" ]; then
  echo "错误：缺少视频文件参数！"
  echo "使用方法: ./big_loop.sh <video_name>"
  exit 1
fi
video_name="$1"
cd tram
#python scripts/estimate_camera.py --video "real_videos/$video_name.mov"
python scripts/estimate_camera.py --video "real_videos/$video_name.mov" --static_camera
python scripts/estimate_humans.py --video "real_videos/$video_name.mov"

#python scripts/fixed_root.py $video_name #固定根关节，防止上下飘动

python scripts/visualize_tram.py --video "real_videos/$video_name.mov"

cd ../PHC_x1

python scripts/data_process/npy2npz_convert.py $video_name #把tram输出的npy转换为phc所需的npz

conda activate isaac

python scripts/data_process/fit_smpl_shape.py robot=zhiyuan_x1_fitting_29dof

#输出最终rl训练所需的pkl文件

HYDRA_FULL_ERROR=1 python scripts/data_process/fit_smpl_motion_extra.py robot=zhiyuan_x1_fitting_29dof +amass_root=../ASAPx1/human2humanoid/data/AMASS/AMASS_Compete +input_npz_dir=scripts/data_process/npz_transfered/$video_name +output_pkl_dir=retarget_data +video_name=$video_name