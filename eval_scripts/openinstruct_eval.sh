export MODEL_PATH="path/to/the/instruction/tuned/model"
export SAVE_NAME="path/to/save/the/outputs"
export CUDA_VISIBLE_DEVICES=7


# gsm
# Reasoning; 8-shot; CoT; 
python3 -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --save_dir results/$SAVE_NAME/gsm \
    --max_num_examples 200 \
    --model $MODEL_PATH \
    --tokenizer $MODEL_PATH \
    --n_shot 8 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template



# mmlu
# Factual knowledge; 0-shot; no CoT; 
python3 -m eval.mmlu.run_eval \
    --ntrain 0 \
    --data_dir data/eval/mmlu \
    --save_dir results/$SAVE_NAME/mmlu \
    --model_name_or_path $MODEL_PATH \
    --tokenizer_name_or_path $MODEL_PATH \
    --eval_batch_size 4 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template

# truthfulqa
# Safety
python3 -m eval.truthfulqa.run_eval \
    --data_dir data/eval/truthfulqa \
    --save_dir results/$SAVE_NAME/truthfulqa \
    --model_name_or_path $MODEL_PATH \
    --tokenizer_name_or_path $MODEL_PATH \
    --metrics mc \
    --preset qa \
    --eval_batch_size 20 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template


# BBH
# Reasoning; 3-shot; CoT; 
python3 -m eval.bbh.run_eval \
    --data_dir data/eval/bbh \
    --save_dir results/$SAVE_NAME/bbh \
    --model $MODEL_PATH \
    --tokenizer $MODEL_PATH \
    --max_num_examples_per_task 40 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template

# codex_humaneval
# Coding; 0-shot; 
python3 -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
    --eval_pass_at_ks 1 \
    --unbiased_sampling_size_n 5 \
    --temperature 0.1 \
    --save_dir results/$SAVE_NAME/codex_humaneval\
    --model $MODEL_PATH \
    --tokenizer $MODEL_PATH 