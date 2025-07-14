#!/bin/bash

# 定义学习率和数据类型的列表
learning_rates=("1e-3" "5e-4" "1e-4" "5e-5" "1e-5")
data_types=("A" "V")

# 遍历学习率和数据类型的所有组合
for lr in "${learning_rates[@]}"; do
  for dt in "${data_types[@]}"; do
    # 执行实验并传递参数
    echo "Running experiment with learning_rate=$lr and data_type=$dt"
    python ./code/train_student.py --learning_rate $lr --data_type $dt
  done
done
