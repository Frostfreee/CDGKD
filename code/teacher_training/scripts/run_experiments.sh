#!/bin/bash

# 设置一些通用参数
batch_size=128
epochs=100
# pmr_loss="False"
# contrastive_learning="False"
# data_type="A"
learning_rate=1e-4

# 定义多个实验的配置
declare -a data_types=("A" "V" )      # 数据类型
# declare -a distance_types=("euclidean" "cosine")  # 距离度量
# declare -a learning_rates=(1e-5 5e-5 1e-4)
declare -a alphas=(0.5 1 1.5 2 2.5 3)
# declare -a batch_sizes=(64 128)

# 遍历所有组合并运行实验
for data_type in "${data_types[@]}"; do
    for alpha in "${alphas[@]}"; do
        echo "运行实验： Data_type=$data_type alpha=$alpha" 

            # 执行 Python 脚本并传入相应的参数
            python ./teacher_training/training/train_teacher.py \
            --batch_size=$batch_size \
            --epoch=$epochs \
            --learning_rate=$learning_rate \
            --data_type="$data_type" \
            --alpha=$alpha \
            # --contrastive_learning=$contrastive_learning
    done
done
