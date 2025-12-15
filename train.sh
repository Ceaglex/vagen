
export PYTHONPATH=$PYTHONPATH:$(pwd)
export TOKENIZERS_PARALLELISM=false
# # export OMP_NUM_THREADS=28
# export http_proxy=http://star-proxy.oa.com:3128
# export https_proxy=http://star-proxy.oa.com:3128


GPU_IDS="0,1,2,3"
GPU_NUMS=4


: "
    For ttv:
"
##### ttv training code may need modified
# CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch --num_processes=$GPU_NUMS \
#     --main_process_port 29512 \
#     worker/text_to_video_base_wan.py \
#     --config config/ttv_wan.yaml 
##### ttv code may need modified


: "
    For tta:
"

##### tta flow final version
# CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch --num_processes=$GPU_NUMS \
#     --main_process_port 29522 \
#     worker/text_to_audio_base_sd.py \
#     --config config/tta_sd.yaml 
# CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch --num_processes=$GPU_NUMS \
#     --main_process_port 29532 \
#     worker/text_to_audio_base_sd.py \
#     --config config/tta_sd_ft.yaml 
##### tta flow final version

# CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch --num_processes=$GPU_NUMS \
#     --main_process_port 29542 \
#     worker/text_to_audio_base_wan.py \
#     --config config/tta_wan.yaml 




: "
    For ttva:
"
# CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch --num_processes=$GPU_NUMS \
#     --main_process_port 29501 \
#     worker/text_to_video_audio_base_wan_sd.py \
#     --config config/ttva_wan_sd.yaml 



 CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch --num_processes=$GPU_NUMS \
    --main_process_port 29503 \
    worker/text_to_video_audio_sft_base_ovi.py \
    --config config/ttva_sft_ovi.yaml 
    

: "
    For ttva dpo:
"

# CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch --num_processes=$GPU_NUMS \
#     --main_process_port 29504 \
#     worker/text_to_video_audio_dpo_base_wan_sd.py \
#     --config config/ttva_dpo_wan_sd.yaml 

 

# # # #  CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch --num_processes=$GPU_NUMS \
# # # #     --main_process_port 29503 \
# # # #     worker/text_to_video_audio_dpo_dsft_base_ovi.py \
# # # #     --config config/ttva_dpo_dsft_ovi.yaml 

