export CUDA_VISIBLE_DEVICES=0

model_name=mixmamba
  python -u run.py \
    --task_name long_term_forecast \
    --model_id ETTh1_96_192 \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 192 \
    --d_model 64 \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --enc_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1 \
    --learning_rate 0.001 \
    --train_epochs 10 \
    --batch_size 32 \
    --num_experts 2 \
    --d_conv 4 \
    --d_state 16 \
    --expand 2 \
    --moe_hidden_factor 2