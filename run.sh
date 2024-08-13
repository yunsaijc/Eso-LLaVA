
# Merge the LoRA weights
python /home/jc/workspace/LLaVA/scripts/merge_lora_weights.py \
       --model-path "/home/jc/workspace/MedLLMs/Eso-Llava/checkpoints/240730v2-lora-llava-med-v1.5-mistral-7b" \
       --model-base "/date/jc/models/MedLLMs/LLaVA-Med/llava-med-v1.5-mistral-7b" \
       --save-model-path "/date/jc/models/MedLLMs/LLaVA-Med/merged/240730v2-lora-llava-med-v1.5-mistral-7b-merged"


# Launch a controller
python -m llava.serve.controller \
    --host 0.0.0.0 \
    --port 10000

# Launch a model worker
# python -m llava.serve.model_worker --host 0.0.0.0 \
#     --controller http://localhost:10000 --port 40000 \
#     --worker http://localhost:40000 \
#     --model-path microsoft/llava-med-v1.5-mistral-7b \
#     --multi-modal
python -m llava.serve.model_worker --host 0.0.0.0 \
    --controller http://localhost:10000 --port 40000 \
    --worker http://localhost:40000 \
    --model-path /home/jc/workspace/MedLLMs/Eso-Llava/checkpoints/240730v2-lora-llava-med-v1.5-mistral-7b-merged \
    --multi-modal

# Launch a gradio web server.
python -m llava.serve.gradio_web_server \
    --controller http://localhost:10000
