#!/usr/bin/env python
set -eux
run() {
    EPOCH=$1
    L_TEMPERATURE=$2
    U_TEMPERATURE=$3
    TRADEOFF=$4
    SEED=$5
    CUDA_VISIBLE_DEVICES=0 python ./kd_atlop_v2_2_gda/run_kd_atlop.py \
        --evaluation_steps 500 \
        --debug 0 \
        --mode train \
        --root_path ./kd_atlop_v2_2_gda \
        --transformer_type bert \
        --model_name_or_path ../model/gaojie/scibert \
        --data_dir ./dataset/gda \
        --train_file train.data \
        --dev_file dev.data \
        --test_file test.data \
        --num_class 2 \
        --train_batch_size 4 \
        --test_batch_size 8 \
        --gradient_accumulation_steps 4 \
        --num_labels 1 \
        --learning_rate 2e-5 \
        --classifier_lr 1e-4 \
        --max_grad_norm 1.0 \
        --warmup_ratio 0.06 \
        --num_train_epochs ${EPOCH} \
        --lower_temperature ${L_TEMPERATURE} \
        --upper_temperature ${U_TEMPERATURE} \
        --loss_tradeoff ${TRADEOFF} \
        --seed ${SEED} \
        --save_ckpt ../output/kd_model.bert-base-cased.epoch_${EPOCH}_ltemp_${L_TEMPERATURE}_utemp_${U_TEMPERATURE}_tradeoff_${TRADEOFF}_seed_${SEED}.ckpt \
        --save_test_pred ../output/kd_result.bert-base-cased.epoch_${EPOCH}_ltemp_${L_TEMPERATURE}_utemp_${U_TEMPERATURE}_tradeoff_${TRADEOFF}_seed_${SEED}.json
}

for epoch in 10; do
  for lower_temperature in 2.0; do
	  for upper_temperature in 20.0; do
      for tradeoff in 5.0 10.0 20.0; do
        for seed in 66 76 86 96 106; do
          run ${epoch} ${lower_temperature} ${upper_temperature} ${tradeoff} ${seed}
		    done
      done
    done
  done
done
