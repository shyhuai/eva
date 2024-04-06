# coding=utf-8
# Copyright (c) 2019-2023 NVIDIA CORPORATION. All rights reserved.
# Copyright 2020 MLBenchmark Group. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language     verning permissions and
# limitations under the License.

"""BERT Pretraining"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import h5py
import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
import logging
import math
import multiprocessing
import numpy as np
import os
import random
import re
import time

from collections import OrderedDict
import itertools
# from concurrent.futures import ProcessPoolExecutor
#from modeling import BertForPretraining, BertConfig
# from schedulers import LinearWarmupPolyDecayScheduler

import utils

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset

from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modeling import BertConfig
from modeling import BertForMaskedLM
from modeling import BertForPretraining
# from schedulers import LinearWarmUpScheduler, LinearWarmupPolyDecayScheduler

# from mlperf_common.frameworks.pyt import PyTProfilerHandler, PyTCommunicationHandler
# from mlperf_logger import mllogger


from fwd_loss_bwd_trainer import preprocess_batch
from torch.cuda.amp import GradScaler
grad_scaler = GradScaler(init_scale=float(os.getenv("INIT_LOSS_SCALE", 2**20)), growth_interval=2000)



# Global variables
skipped_steps = 0
cached_batches = []
last_time_check = 0
def check_sustained_training_time(sustained_training_time, raw_train_start):
    global last_time_check
    if utils.is_main_process():
        if last_time_check == 0 or ((time.time() - raw_train_start) / 60) - last_time_check > 1.0:
            print(f"Training runs {(time.time() - raw_train_start) / 60} mins sustained_training_time {sustained_training_time}")
            last_time_check = (time.time() - raw_train_start) / 60
    return True if sustained_training_time < (time.time() - raw_train_start) / 60 else False

class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed
    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)

def get_eval_batchsize_per_worker(args):
    if torch.distributed.is_initialized():
        chunk_size = args.num_eval_examples // torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        remainder = args.num_eval_examples % torch.distributed.get_world_size()
        if rank<remainder:
            return (chunk_size+1)
        else:
            return chunk_size

def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


def create_eval_dataset(args, worker_init_fn):
    eval_data = []
    for eval_file in sorted(os.listdir(args.eval_dir)):
        eval_file_path = os.path.join(args.eval_dir, eval_file)
        if os.path.isfile(eval_file_path) and 'part' in eval_file_path:
            eval_data.extend(pretraining_dataset(eval_file_path, max_pred_length=args.max_predictions_per_seq, max_seq_length=args.max_seq_length))
            if len(eval_data) > args.num_eval_examples:
                eval_data = eval_data[:args.num_eval_examples]
                break

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                         num_workers=0 if args.eval_batch_size<=10 else 4, worker_init_fn=worker_init_fn, pin_memory=True)
    return eval_dataloader

class pretraining_dataset(Dataset):
    def __init__(self, input_file, max_pred_length, max_seq_length, packed_samples=False, order_samples=False):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        self.max_seq_length = max_seq_length
        self.packed_samples = packed_samples

        f = h5py.File(input_file, "r")
        if not self.packed_samples:
            keys = ['input_ids', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids',
                    'next_sentence_labels']
            self.inputs = [np.asarray(f[key][:]) for key in keys]
        else:
            keys = ['input_ids', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids',
                'packed_input_len', 'packed_masked_lm_len', 'next_sentence_labels', ]
            self.inputs = [np.asarray(f[key][:]) for key in keys]

            if order_samples:
                random_indices = np.arange(len(self.inputs[4]))
                np.random.shuffle(random_indices)
                self.inputs = [feature[random_indices] for feature in self.inputs]

                seq_per_sample = [len(i) for i in self.inputs[4]]
                samples=np.array(list(zip(seq_per_sample,range(len(seq_per_sample)))))
                samples=samples[np.argsort(samples[:,0], kind='stable')]
                sorted_indices=[]
                # fixed for max 3 seq per sample and distribution of 10-1-10 for 1,2,3 seq per sample
                [sorted_indices.extend(j) for i in itertools.zip_longest(
                    batched(samples[samples[:, 0]==1, 1], 10), 
                    batched(samples[samples[:, 0]==2, 1], 1), 
                    batched(samples[samples[:, 0]==3, 1], 10)) for j in i if j!=None ]

                # sorted_indices=np.argsort([len(i) for i in self.inputs[4]], kind='stable')
                self.inputs = [feature[sorted_indices] for feature in self.inputs]        

        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):
        input_ids = np.zeros((self.max_seq_length)).astype(np.int64)
        input_mask= np.zeros((self.max_seq_length)).astype(np.int64)
        segment_ids=np.zeros((self.max_seq_length)).astype(np.int64)
        next_sentence_labels=np.zeros((3)).astype(np.int64)
        packed_input_len = np.zeros((3)).astype(np.int64)

        if not self.packed_samples:
            [_input_ids, _segment_ids, _masked_lm_positions, _masked_lm_ids, _next_sentence_labels] = [
                input[index].astype(np.int64) if indice < 4 else 
                np.asarray(input[index].astype(np.int64)) for indice, input in enumerate(self.inputs)]
        else:
            [_input_ids, _segment_ids, _masked_lm_positions, _masked_lm_ids, _packed_input_len, _packed_masked_lm_len, _next_sentence_labels] = [
                input[index].astype(np.int64) for indice, input in enumerate(self.inputs)]
        
        input_mask_len = _input_ids.shape[-1]
        input_ids[:input_mask_len] = _input_ids
        input_mask[:input_mask_len] = np.ones((1,input_mask_len)).astype(np.int64)        
        segment_ids[:input_mask_len] = _segment_ids
        masked_lm_labels = np.zeros(input_ids.shape, dtype=np.int64)
        masked_lm_labels[ _masked_lm_positions] = _masked_lm_ids

        if not self.packed_samples:
            next_sentence_labels = _next_sentence_labels

            return [torch.from_numpy(input_ids), torch.from_numpy(segment_ids),
                    torch.from_numpy(input_mask), torch.from_numpy(masked_lm_labels), torch.from_numpy(next_sentence_labels)]
        else:
            packed_seqs = _packed_input_len.shape[-1]
            next_sentence_labels[:packed_seqs] = _next_sentence_labels
            packed_input_len[:packed_seqs] = _packed_input_len

            return [torch.from_numpy(input_ids), torch.from_numpy(segment_ids),
                    torch.from_numpy(input_mask), torch.from_numpy(masked_lm_labels), torch.from_numpy(next_sentence_labels),
                    torch.from_numpy(packed_input_len)]

class synthetic_dataset(Dataset):
    def __init__(self, input_file, max_pred_length, max_seq_length, number_of_samples=100):
        self.max_pred_length = max_pred_length
        self.max_seq_length = max_seq_length
        self.samples = []
        self.number_of_samples = number_of_samples
 
        for _ in range(number_of_samples):
            input_ids = np.zeros((self.max_seq_length)).astype(np.int64)
            input_mask= np.zeros((self.max_seq_length)).astype(np.int64)
            segment_ids=np.zeros((self.max_seq_length)).astype(np.int64)
            next_sentence_labels=np.asarray(np.int64(1))
            
            input_mask_len = torch.randint(max_pred_length+1,max_seq_length,(1,))            
            input_ids[:input_mask_len] = torch.randint(2048,30000,(input_mask_len,))
            input_mask[:input_mask_len] = np.ones((1,input_mask_len)).astype(np.int64)
            segment_ids[:input_mask_len] = np.zeros((1,input_mask_len)).astype(np.int64)
            masked_lm_labels = np.zeros(input_ids.shape, dtype=np.int64)
            masked_count = torch.randint(max_pred_length, (1,))
            masked_lm_labels[ torch.randint(max_pred_length, (masked_count,))] = torch.randint(2048,30000,(masked_count,))
            self.samples.append([torch.from_numpy(input_ids), torch.from_numpy(segment_ids),
                    torch.from_numpy(input_mask), torch.from_numpy(masked_lm_labels), torch.from_numpy(next_sentence_labels)])

    def __len__(self):
        return self.number_of_samples*10000

    def __getitem__(self, index):
        return self.samples[index % self.number_of_samples]

def parse_arguments():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_dir",
                        default=None,
                        type=str,
                        help="The input data dir. Should contain .hdf5 files  for the task.")

    parser.add_argument("--packed_samples",
                        default=False,
                        action="store_true",
                        required=False,
                        help="Indicate whether the samples in .hdf5 files contain packed sequences")

    parser.add_argument("--order_samples",
                        default=False,
                        action="store_true",
                        required=False,
                        help="Indicate whether the samples should be ordered by no of sequences 10-1-10, works only with packed_samples")

    parser.add_argument("--max_pack_factor",
                        default=3,
                        type=int,
                        required=False,
                        help="Upto how many sequences can be packed within a sample.")

    parser.add_argument("--average_packing_rate",
                        default=2,
                        type=int,
                        required=False,
                        help="Average number of sequences per batch.")

    parser.add_argument("--synthetic_input",
                        default=False,
                        action='store_true',
                        help="Whether to use synthetic, in-memory dataset - used only in performance measurements")

    parser.add_argument("--bert_model", default="bert-large-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    parser.add_argument("--cuda_graph_mode", default="segmented", type=str,
                        help="'segmented' or 'full_iteration' options for CUDA graph capture. \n"
                        "'segmented' option: Pytorch Autograd orchestrates execution of backward ops every iteration. \n"  
                        "'full_iteration' option: CUDA graph orchestrates execution of bwd ops every iteration without \
                        Autograd involvement (has composability limitations but could be more performant allowing optimizer \
                        and collectives capture).")

    parser.add_argument("--max_iterations_per_graph",
                        default=4,
                        type=int,
                        help="Maximum number of iterations to capture in a single graph. Requires 'full_iteration' option  \
                                for '--cuda_graph_mode'.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        help="The output directory where the model checkpoints will be written.")

    parser.add_argument("--eval_dir",
                        default=None,
                        type=str,
                        help="The eval data dir. Should contain .hdf5 files  for the task.")
    parser.add_argument("--eval_iter_start_samples",
                        default=0,
                        type=int,
                        help="Sample to begin performing eval.")
    parser.add_argument("--eval_iter_samples",
                        default=-1,
                        type=int,
                        help="If set to -1, disable eval, \
                        else evaluate every eval_iter_samples during training")
    parser.add_argument("--num_eval_examples",
                        default=10000,
                        type=int,
                        help="number of eval examples to run eval on")
    parser.add_argument("--cache_eval_data",
                        default=False,
                        action='store_true',
                        help="whether to cache evaluation data on GPU")
    parser.add_argument("--load_eval_synchronously",
                        default=False,
                        action='store_true',
                        help="whether to force synchronous load of eval set (needed for 1gpu config)")

    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="The initial checkpoint to start training from.")
    parser.add_argument("--init_tf_checkpoint",
                        default=None,
                        type=str,
                        help="The initial TF checkpoint to start training from.")

    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_predictions_per_seq",
                        default=76,
                        type=int,
                        help="The maximum total of masked tokens in input sequence")
    parser.add_argument("--train_batch_size",
                        default=18,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=128,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=4e-5,
                        type=float,
                        help="The initial learning rate for LAMB.")
    parser.add_argument("--weight_decay_rate",
                        default=0.01,
                        type=float,
                        help="weight decay rate for LAMB.")
    parser.add_argument("--opt_lamb_beta_1",
                        default=0.9,
                        type=float,
                        help="LAMB beta1.")
    parser.add_argument("--opt_lamb_beta_2",
                        default=0.999,
                        type=float,
                        help="LAMB beta2.")
    parser.add_argument("--max_steps",
                        default=1536,
                        type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--sustained_training_time",
                        '-stt',
                        type=int,
                        default=0,
                        help="Total training time")
    parser.add_argument("--max_samples_termination",
                        default=14000000,
                        type=float,
                        help="Total number of training samples to run.")
    parser.add_argument("--warmup_proportion",
                        default=0.01,
                        type=float,
                        help="Proportion of optimizer update steps to perform linear learning rate warmup for. "
                             "Typically 1/8th of steps for Phase2")
    parser.add_argument("--warmup_steps",
                        default=0,
                        type=float,
                        help="Number of optimizer update steps to perform linear learning rate warmup for. "
                             "Typically 1/8th of steps for Phase2")
    parser.add_argument("--start_warmup_step",
                        default=0,
                        type=float,
                        help="Starting step for warmup. ")
    parser.add_argument("--local_rank",
                        type=int,
                        default=os.getenv('LOCAL_RANK', 0),
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0.0,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--log_freq',
                        type=float, default=1.0,
                        help='frequency of logging loss. If not positive, no logging is provided for training loss')
    parser.add_argument('--checkpoint_activations',
                        default=False,
                        action='store_true',
                        help="Whether to use gradient checkpointing")
    parser.add_argument("--resume_from_checkpoint",
                        default=False,
                        action='store_true',
                        help="Whether to resume training from checkpoint. If set, precedes init_checkpoint/init_tf_checkpoint")
    parser.add_argument('--keep_n_most_recent_checkpoints',
                        type=int,
                        default=20,
                        help="Number of checkpoints to keep (rolling basis).")
    parser.add_argument('--num_samples_per_checkpoint',
                        type=int,
                        default=500000,
                        help="Number of update steps until a model checkpoint is saved to disk.")
    parser.add_argument('--min_samples_to_start_checkpoints',
                        type=int,
                        default=3000000,
                        help="Number of update steps until model checkpoints start saving to disk.")
    parser.add_argument('--skip_checkpoint',
                        default=False,
                        action='store_true',
                        help="Whether to save checkpoints")
    parser.add_argument('--phase2',
                        default=False,
                        action='store_true',
                        help="Only required for checkpoint saving format")
    parser.add_argument('--allreduce_post_accumulation',
                        default=False,
                        action='store_true',
                        help="Whether to do allreduces during gradient accumulation steps.")
    parser.add_argument('--allreduce_post_accumulation_fp16',
                        default=False,
                        action='store_true',
                        help="Whether to do fp16 allreduce post accumulation.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--exchange_padding",
                        default=False,
                        action='store_true',
                        help="Whether to run with unpadding.")
    parser.add_argument("--unpad",
                        default=False,
                        action='store_true',
                        help="Whether to run with unpadding.")
    parser.add_argument("--unpad_fmha",
                        default=False,
                        action='store_true',
                        help="Whether to run fmha with unpadding.")
    parser.add_argument("--pad_fmha",
                        default=False,
                        action='store_true',
                        help="Whether to run fmha with padding.")
    parser.add_argument("--pad",
                        default=False,
                        action='store_true',
                        help="Whether to pad tokens.")
    parser.add_argument("--enable_fuse_dropout",
                        default=False,
                        action='store_true',
                        help="Whether to disable fusion of attention mask to softmax and dropout.")
    parser.add_argument("--disable_fuse_mask",
                        default=False,
                        action='store_true',
                        help="Whether to disable fusion of the attention mask to softmax.")
    parser.add_argument("--disable_fuse_scale",
                        default=False,
                        action='store_true',
                        help="Whether to disable fusion of the scaling to BMM1.")
    parser.add_argument("--disable_fuse_qkv",
                        default=False,
                        action='store_true',
                        help="Whether to disable fusion of the QKV GEMMs.")
    parser.add_argument("--disable_apex_softmax",
                        default=False,
                        action='store_true',
                        help="Whether to disable apex softmax.")
    parser.add_argument("--enable_stream",
                        default=False,
                        action='store_true',
                        help="Enable use of streams for pad case.")
    parser.add_argument("--fused_gemm_gelu",
                        default=False,
                        action='store_true',
                        help="Whether to fuse gemm and gelu together.")
    parser.add_argument("--fused_mha",
                        default=False,
                        action='store_true',
                        help="Whether to run with optimizations.")
    parser.add_argument("--no-fused_mha",
                        dest='fused_mha',
                        action='store_false',
                        help="Disable fused MHA optimization.")
    parser.add_argument("--fused_gelu_bias",
                        default=False,
                        action='store_true',
                        help="Whether to run with optimizations.")
    parser.add_argument("--fused_dropout_add",
                        default=False,
                        action='store_true',
                        help="Whether to run with optimizations.")
    parser.add_argument("--fused_bias_mha",
                        default=False,
                        action='store_true',
                        help="Whether to run with optimizations.")
    parser.add_argument("--fused_bias_fc",
                        default=False,
                        action='store_true',
                        help="Whether to run with optimizations.")
    parser.add_argument("--fused_bias_fc_loss_head",
                        default=False,
                        action='store_true',
                        help="Whether to run with optimizations.")
    parser.add_argument("--dense_seq_output",
                        default=False,
                        action='store_true',
                        help="Whether to run with optimizations.")
    parser.add_argument("--use_env",
                        action='store_true',
                        help="Whether to read local rank from ENVVAR")
    parser.add_argument('--bert_config_path',
                        type=str,
                        default="/workspace/phase1",
                        help="Path bert_config.json is located in")
    parser.add_argument('--target_mlm_accuracy',
                        type=float,
                        default=0.0,
                        help="Stop training after reaching this Masked-LM accuracy")
    parser.add_argument('--train_mlm_accuracy_window_size',
                        type=int,
                        default=0,
                        help="Average accuracy over this amount of batches before performing a stopping criterion test")
    parser.add_argument('--num_epochs_to_generate_seeds_for',
                        type=int,
                        default=2,
                        help="Number of epochs to plan seeds for. Same set across all workers.")

    parser.add_argument("--use_cuda_graph",
                        default=False,
                        action='store_true',
                        help="Enable CUDA graph execution.")
    parser.add_argument("--use_ddp",
                        default=False,
                        action='store_true',
                        help="Enable DDP.")
    parser.add_argument("--ddp_type",
                        default='apex',
                        type=str,
                        help="DDP type: 'apex' or 'native'.")
    parser.add_argument("--use_gradient_as_bucket_view",
                        default=False,
                        action='store_true',
                        help="Turn ON gradient_as_bucket_view optimization in native DDP.")
    parser.add_argument("--bypass_amp",
                        default=False,
                        action='store_true',
                        help="Bypass AMP unscaling and inf/nan checks for SOL measurements.")
    parser.add_argument('--distributed_lamb',
                        default=False,
                        action='store_true',
                        help="Whether to use distributed lamb.")
    parser.add_argument('--dwu-group-size', '--dwugs',
                        default=0,
                        type=int,
                        metavar='DWUGS',
                        help='distributed weight update group size. If arg is 0, defaults to one node')
    parser.add_argument('--dwu-num-blocks',
                        '--dwunb',
                        default=1,
                        type=int,
                        metavar='DWUNB',
                        help='number of blocks in dwu scheme')
    parser.add_argument('--dwu-num-chunks',
                        '--dwunc',
                        default=1,
                        type=int,
                        metavar='DWUNC',
                        help='number of chunks in dwu scheme')
    parser.add_argument('--dwu-num-rs-pg',
                        '--dwurspg',
                        default=2,
                        type=int,
                        metavar='DWURSPG',
                        help='number of reduction-scatter streams in dwu scheme')
    parser.add_argument('--dwu-num-ar-pg',
                        '--dwuarpg',
                        default=4,
                        type=int,
                        metavar='DWUARPG',
                        help='number of all-reduce streams in dwu scheme')
    parser.add_argument('--dwu-num-ag-pg',
                        '--dwuagpg',
                        default=2,
                        type=int,
                        metavar='DWUAGPG',
                        help='number of all-gather streams in dwu scheme')
    parser.add_argument('--dwu-overlap-reductions',
                        default=False,
                        action='store_true',
                        help='whether to overlap reductions with backprop')
    parser.add_argument('--dwu-e5m2-allgather',
                        default=False,
                        action='store_true',
                        help='do allgather with e5m2 floats')
    parser.add_argument('--use_transformer_engine2',
                        default=False,
                        action='store_true',
                        help='Enable FP8 layers')
    args = parser.parse_args()

    return args



def remap_parameters(model_dict, config):
    res_dict = OrderedDict()
    for k in model_dict:
        if 'dense_act' in k:
            new_k = k.replace('dense_act', 'dense')
        else:
            new_k = k
        res_dict[new_k] = model_dict[k]
    model_dict.clear()
    return res_dict

def prepare_model_and_optimizer(args,device):
    global_step = 0
    args.resume_step = 0
    checkpoint = None

    config = BertConfig.from_json_file(args.bert_config_path)
    config.fused_mha = args.fused_mha
    config.fused_gelu_bias = args.fused_gelu_bias
    config.fused_bias_mha = args.fused_bias_mha
    config.fused_bias_fc = args.fused_bias_fc
    config.fused_bias_fc_loss_head = args.fused_bias_fc_loss_head
    config.fused_gemm_gelu = args.fused_gemm_gelu
    config.dense_seq_output = args.dense_seq_output
    config.unpad = args.unpad
    config.unpad_fmha = args.unpad_fmha
    config.pad_fmha = args.pad_fmha
    config.max_seq_length = args.max_seq_length
    config.pad = args.pad
    config.fuse_qkv = not args.disable_fuse_qkv
    config.fuse_scale = not args.disable_fuse_scale
    config.fuse_mask = not args.disable_fuse_mask
    config.fuse_dropout = args.enable_fuse_dropout
    config.fused_dropout_add = args.fused_dropout_add
    config.apex_softmax = not args.disable_apex_softmax
    config.enable_stream = args.enable_stream
    if config.fuse_mask == True: config.apex_softmax = True
    if config.pad == False: config.enable_stream = True
    if config.unpad == True: config.fused_mha = False
    config.packed_samples = args.packed_samples
    # FP8 recipe
    config.use_transformer_engine2 = args.use_transformer_engine2

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    # Load from Pyt checkpoint - either given as init_checkpoint, or picked up from output_dir if found
    if args.init_checkpoint is not None:
        # Prepare model
        #model = BertForPreTraining(config)
        model = BertForPretraining(config)

        # for k,v in model.state_dict().items():
        #     print(f'model-k,len(v)={k}, {v.numel()}')

        checkpoint=torch.load(args.init_checkpoint, map_location="cpu")["model"]
        checkpoint_remap = remap_parameters(checkpoint, config)
        model.load_state_dict(checkpoint_remap, strict=True)
    
    model.to(device=device)
    return model, checkpoint, global_step        



def run_eval(args, model, eval_dataloader, device):
    model.eval()

    total_eval_loss, total_eval_mlm_acc = 0.0, 0.0
    total_masked = 0.0

    with torch.no_grad():
        for batch in eval_dataloader:
            # batch = preprocess_batch(args, *batch)
            batch = [t.to(device) for t in batch]
            loss, mlm_acc, num_masked = model(*batch)
            # print(loss,mlm_acc,num_masked)
            total_eval_loss += loss * num_masked
            total_eval_mlm_acc += mlm_acc * num_masked
            total_masked += num_masked

    # Average by number of examples
    total_eval_mlm_acc /= total_masked
    total_eval_loss /= total_masked

    return total_eval_loss.item(), total_eval_mlm_acc.item()


def main():
    args = parse_arguments()
    status = 'aborted'  # later set to 'success' if termination criteria met
    # mllogger.start(key=mllogger.constants.INIT_START, sync=False)
    # if args.use_env and 'LOCAL_RANK' in os.environ:
    #     args.local_rank = int(os.environ['LOCAL_RANK'])

    if utils.is_main_process():
        #print("parsed args:")
        #print(args)
        pass
    current_device = torch.device('cuda', torch.cuda.current_device())
    model, checkpoint, global_step = prepare_model_and_optimizer(args,current_device)


    final_loss = float("inf")
    train_time_raw = float("inf")
    raw_train_start = time.time()

    if args.do_train:

        worker_init = WorkerInitObj(0)
        # Start prefetching eval dataset
        if args.eval_dir:
            eval_dataloader = create_eval_dataset(args, worker_init_fn=worker_init)
            # print(len(eval_dataloader))
            eval_avg_loss, eval_avg_mlm_accuracy = run_eval(args, model, eval_dataloader, current_device)

            if utils.is_main_process():
                print({"ptfile:": args.init_checkpoint, "global_steps": global_step, "eval_loss": eval_avg_loss, "eval_mlm_accuracy":eval_avg_mlm_accuracy})

        
    return args, final_loss, train_time_raw

if __name__ == "__main__":
    #torch.backends.cuda._stateful_ops.state_on_device = True

    now = time.time()
    args, final_loss, train_time_raw = main()


