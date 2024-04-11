ptfile="${ptfile:-lamb_ckpt_20001_32000.pt}"
RESULT_PATH=/datasets/results/pretrain_ckpts
NCCL_P2P_DISABLE=1 python /workspace/repos/deva/examples/bert/eval/run_pretraining_only_eval.py  \
 --eval_dir /share/datasets/bert/bert_large_eval_data/hdf5/eval_varlength/ \
 --init_checkpoint  $RESULT_PATH/$ptfile \
 --max_seq_length 512 \
 --do_train \
 --bert_config_path /workspace/repos/deva/examples/bert/eval/config/bert_config.json

# --init_checkpoint  /datasets/bert_pretrain_ckpts/eva_ckpt_8000_8000.pt \
