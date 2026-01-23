#!/usr/bin/env bash

SESSION_NAME="arian-servers"
CONDA_ENV="arian-venv"

tmux kill-session -t $SESSION_NAME 2>/dev/null
tmux new-session -d -s $SESSION_NAME -n $SESSION_NAME

# Global Settings
tmux set -g pane-border-status top
tmux set -g pane-border-format " [ #T ] "

SETUP_CMD="source \$(conda info --base)/etc/profile.d/conda.sh && conda activate $CONDA_ENV && export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo"

tmux select-pane -t $SESSION_NAME:0.0 -T "Qwen14-LLM"

tmux split-window -v -t $SESSION_NAME:0.0 
tmux split-window -h -t $SESSION_NAME:0.1
# tmux split-window -h -t $SESSION_NAME:0.0

tmux select-pane -t $SESSION_NAME:0.1 -T "Qwen3-LLM"
tmux select-pane -t $SESSION_NAME:0.2 -T "BGE-Reranker"
#tmux select-pane -t $SESSION_NAME:0.3 -T "Deepseek-OCR"

# Pane 0.0: Qwen14 (GPUs 0,1)
tmux send-keys -t $SESSION_NAME:0.0 "$SETUP_CMD" C-m
tmux send-keys -t $SESSION_NAME:0.0 "CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8 --tensor-parallel-size 2 --gpu-memory-utilization 0.8 --port 8001 --dtype float16 --max-model-len 32768" C-m

# Pane 0.1: Qwen3 (GPU 3)
tmux send-keys -t $SESSION_NAME:0.1 "$SETUP_CMD" C-m
tmux send-keys -t $SESSION_NAME:0.1 "CUDA_VISIBLE_DEVICES=3 vllm serve Qwen/Qwen2.5-3B-Instruct --port 8002 --dtype float16 --gpu-memory-utilization 0.7 --max-model-len 17000 --enable-auto-tool-choice --tool-call-parser hermes" C-m

#Pane 0.2: BGE-Reranker (GPU 2)
tmux send-keys -t $SESSION_NAME:0.2 "$SETUP_CMD" C-m
tmux send-keys -t $SESSION_NAME:0.2 "CUDA_VISIBLE_DEVICES=2 vllm serve BAAI/bge-reranker-v2-m3 --port 8003 --dtype float16 --gpu-memory-utilization 0.3 --pooler-config.use_activation true" C-m

# Pane 0.3: Deepseek OCR (GPU 2)
# tmux send-keys -t $SESSION_NAME:0.3 "$SETUP_CMD" C-m
# tmux send-keys -t $SESSION_NAME:0.3 "CUDA_VISIBLE_DEVICES=2 vllm serve deepseek-ai/DeepSeek-OCR --port 8004 --dtype float16 --gpu-memory-utilization 0.6 --max-model-len 4096 --logits-processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor --mm-processor-cache-gb 0 --no-enable-prefix-caching" C-m

tmux attach-session -t $SESSION_NAME
