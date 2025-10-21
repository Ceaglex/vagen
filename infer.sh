
export PYTHONPATH=$PYTHONPATH:$(pwd)
export TOKENIZERS_PARALLELISM=false
# # export OMP_NUM_THREADS=28
# export http_proxy=http://star-proxy.oa.com:3128
# export https_proxy=http://star-proxy.oa.com:3128


GPU_IDS="0,1,2,3,4,5,6,7"
GPU_NUMS=8




: "
    For ttva:
"


# CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch --num_processes=$GPU_NUMS \
#     --main_process_port 29506 \
#     worker/infer_wan_sd.py \
#     --config config/infer_wan_sd.yaml 


CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch --num_processes=$GPU_NUMS \
    --main_process_port 29506 \
    worker/infer_ovi.py \
    --config config/infer_ovi.yaml 