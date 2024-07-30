
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
    --model-base microsoft/llava-med-v1.5-mistral-7b \
    --model-path /home/jc/workspace/MedLLMs/Eso-Llava/checkpoints/240730v1 \
    --multi-modal

# Launch a gradio web server.
python -m llava.serve.gradio_web_server \
    --controller http://localhost:10000
