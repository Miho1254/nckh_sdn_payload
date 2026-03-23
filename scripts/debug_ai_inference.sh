#!/bin/bash
# Script debug AI inference - kiểm tra xem AI thread có chạy không

echo "=== DEBUG AI INFERENCE ==="

# Kiểm tra xem AI thread có được khởi tạo không
docker exec nckh-sdn-mininet python3 -c "
import sys
sys.path.insert(0, '/work')
from controller_stats import FatTreeController
from ryu.base import app_manager

# Check if AI is available
try:
    import torch
    AI_AVAILABLE = True
    print(f'PyTorch available: {AI_AVAILABLE}')
    print(f'CUDA available: {torch.cuda.is_available()}')
except ImportError:
    AI_AVAILABLE = False
    print(f'PyTorch available: {AI_AVAILABLE}')

# Check if AC model is available
try:
    from ai_model.tft_ac_net import TFT_ActorCritic_Model
    AC_AVAILABLE = True
    print(f'TFT-AC available: {AC_AVAILABLE}')
except ImportError:
    AC_AVAILABLE = False
    print(f'TFT-AC available: {AC_AVAILABLE}')

# Try to load model
if AI_AVAILABLE and AC_AVAILABLE:
    try:
        from config import BACKENDS
        from ai_model.tft_ac_net import TFT_ActorCritic_Model
        
        ckpt = torch.load('/work/ai_model/checkpoints/tft_ac_best.pth', map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        
        vsn_key = 'vsn.weight_grn.fc1.weight'
        input_size = state_dict[vsn_key].shape[1] if vsn_key in state_dict else 42
        
        hidden_key = 'lstm.weight_ih_l0'
        hidden_size = state_dict[hidden_key].shape[0] // 4 if hidden_key in state_dict else 64
        
        model = TFT_ActorCritic_Model(
            input_size=input_size, seq_len=5, hidden_size=hidden_size, num_actions=len(BACKENDS)
        )
        model.load_state_dict(state_dict)
        model.eval()
        
        print(f'Model loaded successfully!')
        print(f'Input size: {input_size}, Hidden size: {hidden_size}')
        
        # Test inference
        import numpy as np
        state_buffer = [[0.0] * input_size for _ in range(5)]
        import torch
        state_tensor = torch.tensor([state_buffer], dtype=torch.float32)
        probs = model.get_policy(state_tensor)
        print(f'Policy probs: {probs}')
        
    except Exception as e:
        print(f'Error loading model: {e}')
"

echo ""
echo "=== CHECK CONTROLLER LOGS ==="
docker exec nckh-sdn-mininet cat /tmp/ryu.log 2>/dev/null | tail -50

echo ""
echo "=== CHECK INFERENCE LOG ==="
docker exec nckh-sdn-mininet cat /work/stats/inference_log.csv 2>/dev/null | head -20
