"""
SMPL-IKS API quick test.

Verifies that the basic API functionality works.

Usage (run from the project root):
    python third_parties/api/test_smpliks.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from third_parties.api.smpliks import smpliks

print("Testing the SMPL-IKS API...")
print("=" * 50)

# Create test data [B, 24, 3]
batch_size = 2
skeleton = torch.randn(batch_size, 24, 3).cuda()

print(f"Input: {skeleton.shape}")

# Call API
theta, beta = smpliks(skeleton)

print(f"Output theta: {theta.shape}")  # [B, 72]
print(f"Output beta: {beta.shape}")    # [B, 10]

print("\n✓ API test passed!")
print("=" * 50)
