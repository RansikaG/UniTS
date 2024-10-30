model_name=UniTS # Model name, UniTS
exp_name=UniTS_supervised_x64 # Exp name
wandb_mode=online # Use wandb to log the training, change to disabled if you don't want to use it
project_name=supervised_learning # preject name in wandb

random_port=$((RANDOM % 9000 + 1000))
#  --pretrained_weight /home/ransika/UniTS/newcheckpoints/units_x64_supervised_checkpoint.pth \

# Supervised learning
torchrun --nnodes 1 --nproc-per-node=1  --master_port $random_port  /home/ransika/UniTS/run.py \
  --is_training 1 \
  --model_id $exp_name \
  --model $model_name \
  --prompt_num 10 \
  --patch_len 4 \
  --stride 4 \
  --e_layers 4 \
  --d_model 64 \
  --des 'Exp' \
  --learning_rate 2e-5 \
  --weight_decay 1e-5 \
  --lradj one_cycle \
  --train_epochs 50 \
  --batch_size 32 \
  --acc_it 32 \
  --debug $wandb_mode \
  --project_name $project_name \
  --clip_grad 100 \
  --task_data_config_path /home/ransika/UniTS/traffic_data/classification_task.yaml