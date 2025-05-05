#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
#swanlab 环境模块
export SWANLAB_HOST=https://swanlab.cn
export SWANLAB_API_KEY=JtEPVvIMaPOLhVbHXivVl


#BSUB -q gpu_v100             # 指定使用 gpu_v100 队列
#BSUB -J deepseek_train            # 定义作业名称（可自定义）
#BSUB -gpu "num=4:mode=shared"            # 定义使用的 GPU 数量；如需要多块GPU可修改，如 "num=4"
#BSUB -n 20

#BSUB -o /share/home/zhangshanqi/pyn/shopping_behavior2/models/pretrained/output/train.out        # 标准输出文件，%J会替换为作业号
#BSUB -e /share/home/zhangshanqi/pyn/shopping_behavior2/models/pretrained/output/train.err              # 错误输出文件


# 设置MPI对接环境（如作业不需要MPI，可注释掉或保留以防后续扩展）
export I_MPI_HYDRA_BOOTSTRAP=lsf

# 载入必要的环境模块（根据系统情况调整）
source /share/apps/intel/intel_2017u4/compilers_and_libraries_2017.4.196/linux/bin/compilervars.sh intel64
source /share/apps/intel/intel_2017u4/compilers_and_libraries_2017.4.196/linux/mkl/bin/mklvars.sh intel64

# 设置CUDA相关环境变量（确保能使用GPU加速库，如TensorFlow、PyTorch等）
export PATH=/share/gpu-apps/cuda/12.2/bin:$PATH
export LD_LIBRARY_PATH=/share/gpu-apps/cuda/12.2/lib64:$LD_LIBRARY_PATH
export PATH=~/pytorch/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/gcc/gcc10.2/lib64:$LD_LIBRARY_PATH
# 进入作业文件所在目录
cd /share/home/zhangshanqi/pyn/shopping_behavior2/models/pretrained




 #执行 Python 脚本，确保 train.py 中正确调用了GPU加速相关的
deepspeed --num_gpus=4 training_swanlab.py \
  --deepspeed ds_config.json \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8

 
 

