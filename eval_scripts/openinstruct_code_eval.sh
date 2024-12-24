export MODEL_PATH="path/to/the/instruction/tuned/model"
export SAVE_NAME="path/to/save/the/outputs"
export CUDA_VISIBLE_DEVICES=0


# codex_humaneval
# Coding; 0-shot; 
python3 -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
    --eval_pass_at_ks 1 5 10 \
    --unbiased_sampling_size_n 10 \
    --temperature 0.1 \
    --save_dir results/$SAVE_NAME/codex_humaneval\
    --model $MODEL_PATH \
    --tokenizer $MODEL_PATH 

# mbpp
# Coding;
export HF_ALLOW_CODE_EVAL=1
python3 -m eval.mbpp.run_eval \
    --eval_pass_at_ks 1 5 10 \
    --unbiased_sampling_size_n 10 \
    --temperature 0.1 \
    --save_dir results/$SAVE_NAME/mbpp \
    --model $MODEL_PATH \
    --tokenizer $MODEL_PATH \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
    --additional_stop_sequence ' ```'