
export PYTHONPATH=$PYTHONPATH:$(pwd)
export TOKENIZERS_PARALLELISM=false
# # export OMP_NUM_THREADS=28
# export http_proxy=http://star-proxy.oa.com:3128
# export https_proxy=http://star-proxy.oa.com:3128


GPU_IDS="0,1,2,3,4,5,6,7"
GPU_NUMS=4


: "
    For ttv:
"
# CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch --num_processes=$GPU_NUMS \
#     --main_process_port 29502 \
#     worker/text_to_video_base_wan.py \
#     --config config/ttv_wan.yaml 


: "
    For tta:
"


CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch --num_processes=$GPU_NUMS \
    --main_process_port 29506 \
    worker/infer.py \
    --config config/infer.yaml 


