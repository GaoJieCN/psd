#!/usr/bin/env python
set -eux
run() {
    EPOCH=$1
    L_TEMPERATURE=$2
    U_TEMPERATURE=$3
    SEED=$4
    CUDA_VISIBLE_DEVICES=0 python ./kd_atlop_v2_2_docred_bert_pair_part/run_kd_atlop.py \
        --debug 0 \
        --mode train \
        --root_path ./kd_atlop_v2_2_docred_bert_pair_part \
        --transformer_type bert \
        --model_name_or_path ../model/gaojie/local-bert-base-cased \
        --data_dir ./dataset/docred \
        --train_file train_annotated.json \
        --dev_file dev.json \
        --valid_file valid.json \
        --test_file test.json \
        --num_class 97 \
        --train_batch_size 4 \
        --test_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --num_labels 4 \
        --learning_rate 3e-5 \
        --classifier_lr 1e-4 \
        --max_grad_norm 1.0 \
        --warmup_ratio 0.06 \
        --num_train_epochs ${EPOCH} \
        --lower_temperature ${L_TEMPERATURE} \
        --upper_temperature ${U_TEMPERATURE} \
        --ikd_loss_tradeoff 20 \
        --rkd_loss_tradeoff 2 \
        --seed ${SEED} \
        --save_ckpt save1 \
        --save_test_pred save2
}

for epoch in 40; do
    for lower_temperature in 2; do
        for upper_temperature in 20; do
            for seed in 96 106; do
                run ${epoch} ${lower_temperature} ${upper_temperature} ${seed}
            done
        done
    done
done