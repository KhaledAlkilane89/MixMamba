export CUDA_VISIBLE_DEVICES=0

model_name=mixmamba


# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/UWaveGestureLibrary/ \
#   --model_id UWaveGestureLibrary \
#   --model $model_name \
#   --data UEA \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model 32 \
#   --learning_rate 0.001 \
#   --batch_size 8 \
#   --num_experts 16 \
#   --d_conv 2 \
#   --d_state 16 \
#   --expand 4 \
#   --moe_hidden_factor 2 \
#   --train_epochs 30 \
#   --patience 10 \


# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/EthanolConcentration/ \
#   --model_id EthanolConcentration \
#   --model $model_name \
#   --data UEA \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model 32 \
#   --learning_rate 0.001 \
#   --batch_size 4 \
#   --num_experts 16 \
#   --d_conv 2 \
#   --d_state 16 \
#   --expand 4 \
#   --moe_hidden_factor 1 \
#   --train_epochs 30 \
#   --patience 10 \

# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/Heartbeat/ \
#   --model_id Heartbeat \
#   --model $model_name \
#   --data UEA \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model 128 \
#   --learning_rate 0.001 \
#   --batch_size 16 \
#   --num_experts 16 \
#   --d_conv 2 \
#   --d_state 16 \
#   --expand 4 \
#   --moe_hidden_factor 1 \
#   --train_epochs 30 \
#   --patience 10 \

# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/JapaneseVowels/ \
#   --model_id JapaneseVowels \
#   --model $model_name \
#   --data UEA \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model 128 \
#   --learning_rate 0.001 \
#   --batch_size 24 \
#   --num_experts 16 \
#   --d_conv 2 \
#   --d_state 16 \
#   --expand 4 \
#   --moe_hidden_factor 1 \
#   --train_epochs 30 \
#   --patience 10 \


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/SelfRegulationSCP1/ \
  --model_id SelfRegulationSCP1 \
  --model $model_name \
  --data UEA \
  --des 'Exp' \
  --itr 1 \
  --d_model 24 \
  --learning_rate 0.001 \
  --batch_size 16 \
  --num_experts 8 \
  --d_conv 2 \
  --d_state 8 \
  --expand 4 \
  --moe_hidden_factor 1 \
  --train_epochs 30 \
  --patience 10 \

# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/SelfRegulationSCP2/ \
#   --model_id SelfRegulationSCP2 \
#   --model $model_name \
#   --data UEA \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model 24 \
#   --learning_rate 0.001 \
#   --batch_size 16 \
#   --num_experts 8 \
#   --d_conv 2 \
#   --d_state 8 \
#   --expand 4 \
#   --moe_hidden_factor 1 \
#   --train_epochs 30 \
#   --patience 10 \


# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/PEMS-SF/ \
#   --model_id PEMS-SF \
#   --model $model_name \
#   --data UEA \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model 32 \
#   --learning_rate 0.001 \
#   --batch_size 32 \
#   --num_experts 64 \
#   --d_conv 2 \
#   --d_state 16 \
#   --expand 4 \
#   --moe_hidden_factor 1 \
#   --train_epochs 30 \
#   --patience 10 \


  # python -u run.py \
  # --task_name classification \
  # --is_training 1 \
  # --root_path ./dataset/FaceDetection/ \
  # --model_id FaceDetection \
  # --model $model_name \
  # --data UEA \
  # --des 'Exp' \
  # --itr 1 \
  # --d_model 24 \
  # --learning_rate 0.001 \
  # --batch_size 32 \
  # --num_experts 8 \
  # --d_conv 2 \
  # --d_state 16 \
  # --expand 4 \
  # --moe_hidden_factor 1 \
  # --train_epochs 30 \
  # --patience 10 \

  # python -u run.py \
  # --task_name classification \
  # --is_training 1 \
  # --root_path ./dataset/Handwriting/ \
  # --model_id Handwriting \
  # --model $model_name \
  # --data UEA \
  # --des 'Exp' \
  # --itr 1 \
  # --d_model 24 \
  # --learning_rate 0.001 \
  # --batch_size 4 \
  # --num_experts 8 \
  # --d_conv 2 \
  # --d_state 16 \
  # --expand 4 \
  # --moe_hidden_factor 1 \
  # --train_epochs 30 \
  # --patience 10 \