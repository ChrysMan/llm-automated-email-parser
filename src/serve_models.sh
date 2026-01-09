#!/usr/bin/env bash

SESSION_NAME="arian_servers"
CONDA_ENV="arian_venv"

# 1. Kill any existing session with this name to start fresh
tmux kill-session -t $SESSION_NAME 2>/dev/null

# 2. Start a new detached session
tmux new-session -d -s $SESSION_NAME -n $SESSION_NAME

# 3. GLOBAL PANE SETTINGS: Enable titles on the top border
tmux set -g pane-border-status top
tmux set -g pane-border-format " [ #T ] " # Shows the title you set

# 4. Define the activation and environment setup
# Use 'source' to ensure conda is initialized correctly in the subshell
SETUP_CMD="source \$(conda info --base)/etc/profile.d/conda.sh && conda activate $CONDA_ENV && export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo"

# 5. Pane 1 (Top): Launch Qwen
# We send the setup command, press Enter (C-m), then send the vLLM command
tmux select-pane -t $SESSION_NAME:0.0 -T "Qwen14-LLM"
tmux send-keys -t $SESSION_NAME:0.0 "$SETUP_CMD" C-m
tmux send-keys -t $SESSION_NAME:0.0 "vllm serve Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8 --tensor-parallel-size 2 --gpu-memory-utilization 0.8 --port 8001 --dtype float16 --max-model-len 22000 --max-num-seqs 10" C-m

# 6. Create Pane 2 (Bottom): Launch Reranker
tmux split-window -v -t $SESSION_NAME:0.0
tmux select-pane -t $SESSION_NAME:0.1 -T "BGE-Reranker"
tmux send-keys -t $SESSION_NAME:0.1 "$SETUP_CMD" C-m
tmux send-keys -t $SESSION_NAME:0.1 "CUDA_VISIBLE_DEVICES=2 vllm serve BAAI/bge-reranker-v2-m3 --api-key dummy --port 8003 --dtype float16 --gpu-memory-utilization 0.4 --tensor-parallel-size 1" C-m

# 7. Create Pane 3 (Right): Launch Qwen3
tmux split-window -h -t $SESSION_NAME:0.1
tmux select-pane -t $SESSION_NAME:0.2 -T "Qwen3-LLM"
tmux send-keys -t $SESSION_NAME:0.2 "$SETUP_CMD" C-m
tmux send-keys -t $SESSION_NAME:0.2 "CUDA_VISIBLE_DEVICES=3 vllm serve Qwen/Qwen2.5-3B-Instruct --port 8002 --dtype float16 --gpu-memory-utilization 0.80 --tensor-parallel-size 1 --max-model-len=17000 --enable-auto-tool-choice --tool-call-parser hermes --chat-template ./utils/tool_chat_template_hermes.jinja" C-m

# 8. Attach to view the dashboard
tmux attach-session -t $SESSION_NAME