export CUDA_VISIBLE_DEVICES=0

model_name=mixmamba

  python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1 \
  --d_model 256 \
  --learning_rate 0.001 \
  --patience 10 \
  --batch_size 16 \
  --num_experts 16 \
  --moe_hidden_factor 2 \
  --patch_len 16 \
  --stride 8 \
  --expand 2 \
  --d_conv 4 \
  --d_state 16 \
  --train_epochs 5 \
  --num_gates 5