#!/usr/bin/env python3
"""
Test V4 Model - Verify action distribution
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/work')
from ai_model.tft_ac_net import TFT_ActorCritic_Model


def test_v4_model():
    """Test V4 model action distribution."""
    
    print("=" * 60)
    print("TESTING V4 MODEL - CAPACITY-WEIGHTED")
    print("=" * 60)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = TFT_ActorCritic_Model(
        input_size=44,
        seq_len=5,
        hidden_size=64,
        num_actions=3
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load('ai_model/checkpoints/tft_ac_v4_best.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']} with accuracy {checkpoint['accuracy']:.2%}")
    print(f"Saved action dist: {checkpoint['action_dist']}")
    
    # Load V4 data
    X = np.load('ai_model/processed_data/X_v4.npy')
    y = np.load('ai_model/processed_data/y_v4.npy')
    
    print(f"\nDataset shape: X={X.shape}, y={y.shape}")
    print(f"Dataset action dist: {np.bincount(y) / len(y)}")
    
    # Test on random samples
    with torch.no_grad():
        X_tensor = torch.tensor(X[:1000], dtype=torch.float32).to(device)
        outputs = model(X_tensor)
        policy = outputs[4]  # safety_probs
        
        # Action predictions
        preds = policy.argmax(dim=1).cpu().numpy()
        action_dist = np.bincount(preds, minlength=3) / len(preds)
        
        print(f"\nModel predictions on 1000 samples:")
        print(f"  Action 0 (h5 - weakest):  {action_dist[0]:.1%}")
        print(f"  Action 1 (h7 - medium):   {action_dist[1]:.1%}")
        print(f"  Action 2 (h8 - strongest): {action_dist[2]:.1%}")
        
        # Test on specific scenarios
        print("\n" + "=" * 60)
        print("SPECIFIC SCENARIO TESTS")
        print("=" * 60)
        
        # Test 1: All zeros (idle state)
        zero_input = torch.zeros(1, 5, 44).to(device)
        out = model(zero_input)
        probs = out[4][0].cpu().numpy()
        print(f"\nIdle state (all zeros):")
        print(f"  Policy: {probs}")
        print(f"  Argmax: {np.argmax(probs)} (should prefer action 2)")
        
        # Test 2: h8 underloaded (should select h8)
        test_input = np.zeros((1, 5, 44))
        test_input[0, -1, 26] = 0.1  # h8 utilization low
        test_input[0, -1, 20] = 0.8  # h5 utilization high
        test_input[0, -1, 23] = 0.5  # h7 utilization medium
        test_tensor = torch.tensor(test_input, dtype=torch.float32).to(device)
        out = model(test_tensor)
        probs = out[4][0].cpu().numpy()
        print(f"\nScenario: h5=80%, h7=50%, h8=10% (h8 underloaded):")
        print(f"  Policy: {probs}")
        print(f"  Argmax: {np.argmax(probs)} (should be action 2 for h8)")
        
        # Test 3: All servers balanced
        test_input = np.zeros((1, 5, 44))
        test_input[0, -1, 26] = 0.5  # h8 utilization
        test_input[0, -1, 20] = 0.5  # h5 utilization
        test_input[0, -1, 23] = 0.5  # h7 utilization
        test_tensor = torch.tensor(test_input, dtype=torch.float32).to(device)
        out = model(test_tensor)
        probs = out[4][0].cpu().numpy()
        print(f"\nScenario: All servers at 50%:")
        print(f"  Policy: {probs}")
        print(f"  Argmax: {np.argmax(probs)} (should prefer action 2 for capacity)")
        
    print("\n" + "=" * 60)
    print("VERIFICATION RESULT")
    print("=" * 60)
    
    # Expected: V4 model should prefer action 2 (h8 - strongest server)
    if action_dist[2] > 0.5:
        print("✓ PASS: Model prefers action 2 (h8 - strongest server)")
        print(f"  Action 2 ratio: {action_dist[2]:.1%}")
    else:
        print("✗ FAIL: Model does not prefer action 2")
        print(f"  Action 2 ratio: {action_dist[2]:.1%}")
    
    return action_dist


if __name__ == '__main__':
    test_v4_model()