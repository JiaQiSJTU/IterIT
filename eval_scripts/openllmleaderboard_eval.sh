export MODEL_PATH="path/to/the/instruction/tuned/model"
export OUTPUT_PATH="path/to/save/the/outputs"

# arc
CUDA_VISIBLE_DEVICES=0 python main.py --model hf-causal-experimental --model_args "pretrained=$MODEL_PATH,use_accelerate=True,dtype=bfloat16" --tasks arc_challenge --num_fewshot 25 --batch_size=1 --output_path "$OUTPUT_PATH/arc_challenge.json"

# hellaswag
CUDA_VISIBLE_DEVICES=0 python main.py --model hf-causal-experimental --model_args "pretrained=$MODEL_PATH,use_accelerate=True,trust_remote_code=True,dtype=bfloat16" --tasks hellaswag --num_fewshot 10 --batch_size=1 --output_path "$OUTPUT_PATH/hellaswag.json"
