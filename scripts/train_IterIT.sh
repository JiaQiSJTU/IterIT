CUDA_VISIBLE_DEVICES=0 python3 src/train.py \
    --model_name_or_path ./Models/Meta-Llama-3-8B \
    --data_path ./data/alpaca_data.json \
    --prompt_template llama-3.1 \
    --bf16 \
    --output_dir ./outputs-alpaca/llama-3-8B-alpaca-IterIT \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 32 \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --report_to tensorboard \
    --seed 42 \
    --update_ratio 3 \
    --update_strategy adaptive \
    --score_type BOTH \
    --ratio 0.05 \
    --diversity_weight_decay 0.1


#   --pre_computed_score_path outputs-alpaca/llama-3-8B-alpaca-IterIT/0_alpaca.jsonl \
