# SMPL-IKS API

A concise API interface for converting 3D skeletons to SMPL parameters.

## API Signature

```python
smpliks(pos_24: torch.Tensor) -> (theta: torch.Tensor, beta: torch.Tensor)
```

## Parameters

- **pos_24** (`torch.Tensor[B, 24, 3]`): 3D skeleton joints
  - SMPL 24-joint topology
  - Units: meters
  - Coordinate system: camera coordinate system (automatically centered on the root joint)
  - Joint order: root, l_hip, r_hip, spine1, l_knee, r_knee, spine2, l_ankle, r_ankle, spine3, l_foot, r_foot, neck, l_collar, r_collar, head, l_shoulder, r_shoulder, l_elbow, r_elbow, l_wrist, r_wrist, l_hand, r_hand

## Return Value

- **theta** (`torch.Tensor[B, 72]`): SMPL pose parameters
  - Axis-angle representation: 24 joints × 3 channels
  - Root joint orientation is valid

- **beta** (`torch.Tensor[B, 10]`): SMPL shape parameters
  - Principal component coefficients of the body shape

## Performance Metrics (3DPW dataset)

| Metric | Value | Official Value |
|------|------|--------|
| MPJPE (joint error) | 0.31mm | 0.20mm |
| MPVPE (vertex error) | 10.73mm | 10.80mm |

## Quick Start

```python
import torch
from third_parties.api.smpliks import smpliks

# Prepare 3D skeleton [B, 24, 3]
skeleton = torch.randn(1, 24, 3).cuda()

# Convert to SMPL parameters
theta, beta = smpliks(skeleton)

print(theta.shape)  # torch.Size([1, 72])
print(beta.shape)   # torch.Size([1, 10])
```

## Method Description

SMPL-IKS adopts a four-stage pipeline:

1. **Shape Inverse (SI)**: Linearly regress β parameters from bone lengths
2. **Pose Refinement (PR-C)**: Analytical method to adjust skeleton orientation
3. **AnalyIK**: Analytical inverse kinematics to solve base θ parameters
4. **MixIK**: Neural network predicts twist angles to refine the pose

## Features

- ✅ Single-function interface, simple and easy to use
- ✅ Automatically initializes on first call (~1 second)
- ✅ Subsequent calls reuse the model for fast inference
- ✅ Supports batch processing
- ✅ GPU acceleration
- ✅ No gradients required (eval mode)

## Notes

- The first call loads the model (~1 second); subsequent calls reuse it
- The input is automatically centered on the root joint
- The model occupies about 100MB of GPU memory
- The input units must be meters

## Testing

Quickly verify API functionality:
```bash
python third_parties/api/test_smpliks.py
```

## Integration Example

```python
import torch
from third_parties.api.smpliks import smpliks

# Your 3D pose estimation results
skeleton_3d = your_3d_pose_estimator(images)  # [B, 24, 3]

# Convert to SMPL parameters
theta, beta = smpliks(skeleton_3d)

# Use SMPL model to generate mesh
from third_parties.smpliks.lib.models.smpl.smpl import SMPLLayer
smpl = SMPLLayer(
    model_path='third_parties/smpliks/IKS/data/smpl/smpliks_data/SMPL_NEUTRAL.pkl',
    kid_template_path='third_parties/smpliks/IKS/data/smpl/smpliks_data/smpl_kid_template.npy',
    dtype=torch.float32,
    age='adult'
).cuda()

mesh = smpl(theta, beta)
```
