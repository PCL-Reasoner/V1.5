#!/bin/bash

set -x

export HCCL_CONNECT_TIMEOUT=3000
export ASCEND_LAUNCH_BLOCKING=1

date=`date '+%Y%m%d_%H%M%S'` 
log_dir=./logs/PCL-Reasoner-V1-RAFT

mkdir -p $log_dir
cp scripts/run_sft.sh scripts/train.sh scripts/tune_qwen25_32b_64k_full_ptd.sh $log_dir

CKPT_LOAD_DIR=/pcl_shared_dpc/fdd/models/hf_sft_packing_0703_step6476/mcore_tp8_pp4/
CKPT_SAVE_DIR=$log_dir
DATA_PATH=/home/yaolu/workspace/Datasets/merged_skywork_R10528_nvidia_57K/mcore/mcore
TOKENIZER_PATH=/home/yaolu/workspace/Models/hf/Qwen2.5-32B

ip_file=/home/yaolu/workspace/MindSpeed-LLM/scripts/node_list.txt

master_ip=`head -n 1 $ip_file`
echo "master IP: $master_ip"
master_port=22333
tot_nodes=`cat $ip_file| wc -l`
echo "total nodes: $tot_nodes"

num_nodes=$tot_nodes
echo "using nodes: $num_nodes"
for((inode=0;inode<$num_nodes;inode++));do
    ip=`head -n $((inode+1)) $ip_file | tail -n 1`
    ssh $ip "cd /home/yaolu/workspace/MindSpeed-LLM && \
        source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
        source /home/yaolu/workspace/miniconda3/bin/activate && \
        conda activate fdd && \
        bash scripts/train.sh \
        $CKPT_LOAD_DIR $CKPT_SAVE_DIR $DATA_PATH $TOKENIZER_PATH\
        $master_ip $master_port $num_nodes $inode" &
done

sleep 6
# loss输出在最后一个节点
echo "tail -f ${CKPT_SAVE_DIR}/node$((num_nodes-1)).log"
tail -f ${CKPT_SAVE_DIR}/node$((num_nodes-1)).log 
