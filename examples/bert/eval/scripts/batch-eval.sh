ptfile=deva_ckpt_801_6000.pt
CUDA_VISIBLE_DEVICES=0 ptfile=$ptfile sh eval/scripts/eval.sh &

ptfile=deva_ckpt_1601_6000.pt
CUDA_VISIBLE_DEVICES=1 ptfile=$ptfile sh eval/scripts/eval.sh & 

ptfile=deva_ckpt_2401_6000.pt
CUDA_VISIBLE_DEVICES=2 ptfile=$ptfile sh eval/scripts/eval.sh &

ptfile=deva_ckpt_3201_6000.pt
CUDA_VISIBLE_DEVICES=3 ptfile=$ptfile sh eval/scripts/eval.sh &

ptfile=deva_ckpt_4001_6000.pt
CUDA_VISIBLE_DEVICES=4 ptfile=$ptfile sh eval/scripts/eval.sh &

ptfile=deva_ckpt_4801_6000.pt
CUDA_VISIBLE_DEVICES=5 ptfile=$ptfile sh eval/scripts/eval.sh &

ptfile=deva_ckpt_5601_6000.pt
CUDA_VISIBLE_DEVICES=6 ptfile=$ptfile sh eval/scripts/eval.sh &

ptfile=deva_ckpt_6000_6000.pt
CUDA_VISIBLE_DEVICES=7 ptfile=$ptfile sh eval/scripts/eval.sh &


