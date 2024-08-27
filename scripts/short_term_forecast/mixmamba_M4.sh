export CUDA_VISIBLE_DEVICES=0

mom=mixmamba

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Yearly' \
  --model_id m4_Yearly \
  --model $mom \
  --data m4 \
  --features M \
  --enc_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --loss 'MASE' \
  --d_model 128 \
  --learning_rate 0.001 \
  --batch_size 16 \
  --num_experts 8 \
  --d_conv 4 \
  --d_state 16 \
  --expand 2 \
  --moe_hidden_factor 2

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Monthly' \
  --model_id m4_Monthly \
  --model $mom \
  --data m4 \
  --features M \
  --enc_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --loss 'MASE' \
   --d_model 128 \
  --learning_rate 0.001 \
  --batch_size 16 \
  --num_experts 8 \
  --d_conv 4 \
  --d_state 16 \
  --expand 2 \
  --moe_hidden_factor 2

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Quarterly' \
  --model_id m4_Quarterly \
  --model $mom \
  --data m4 \
  --features M \
  --enc_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --loss 'MASE' \
   --d_model 128 \
  --learning_rate 0.001 \
  --batch_size 16 \
  --num_experts 8 \
  --d_conv 4 \
  --d_state 16 \
  --expand 2 \
  --moe_hidden_factor 2

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Weekly' \
  --model_id m4_Weekly \
  --model $mom \
  --data m4 \
  --features M \
  --enc_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --loss 'MASE' \
   --d_model 128 \
  --learning_rate 0.001 \
  --batch_size 16 \
  --num_experts 8 \
  --d_conv 4 \
  --d_state 16 \
  --expand 2 \
  --moe_hidden_factor 2

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Daily' \
  --model_id m4_Daily \
  --model $mom \
  --data m4 \
  --features M \
  --enc_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --loss 'MASE' \
   --d_model 128 \
  --learning_rate 0.001 \
  --batch_size 16 \
  --num_experts 8 \
  --d_conv 4 \
  --d_state 16 \
  --expand 2 \
  --moe_hidden_factor 2

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Hourly' \
  --model_id m4_Hourly \
  --model $mom \
  --data m4 \
  --features M \
  --enc_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --loss 'MASE' \
   --d_model 128 \
  --learning_rate 0.001 \
  --batch_size 16 \
  --num_experts 8 \
  --d_conv 4 \
  --d_state 16 \
  --expand 2 \
  --moe_hidden_factor 2
