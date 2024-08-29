export CUDA_VISIBLE_DEVICES=0

model_name=mixmamba

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 \
  --d_model 32 \
  --learning_rate 0.001 \
  --train_epochs 5 \
  --batch_size 16 \
  --num_experts 8 \
  --d_conv 4 \
  --d_state 8 \
  --expand 2 \
  --moe_hidden_factor 2

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model_id ECL_96_192 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 192 \
#   --enc_in 321 \
#   --c_out 321 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model 512 \
#   --learning_rate 0.001 \
#   --train_epochs 5 \
#   --batch_size 16 \
#   --num_experts 8 \
#   --d_conv 4 \
#   --d_state 16 \
#   --expand 2 \
#   --moe_hidden_factor 2

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model_id ECL_96_336 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 336 \
#   --enc_in 321 \
#   --c_out 321 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model 512 \
#   --learning_rate 0.001 \
#   --train_epochs 5 \
#   --batch_size 16 \
#   --num_experts 8 \
#   --d_conv 4 \
#   --d_state 16 \
#   --expand 2 \
#   --moe_hidden_factor 2

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/electricity/ \
#   --data_path electricity.csv \
#   --model_id ECL_96_720 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 720 \
#   --enc_in 321 \
#   --c_out 321 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model 512 \
#   --learning_rate 0.001 \
#   --train_epochs 5 \
#   --batch_size 16 \
#   --num_experts 8 \
#   --d_conv 4 \
#   --d_state 16 \
#   --expand 2 \
#   --moe_hidden_factor 2