"""
Prepare integrated NPZ for mesh evaluation on H36M.

Iterates PKL test metadata -> maps to NPZ prediction -> flattens into:

    prediction:  (T, 17, 3)      H36M 3D joints, camera space, meters
    smpl_param:  dict-like with keys:
        theta:  (T, 72)      GT SMPL pose parameters
        beta:   (T, 10)      GT SMPL shape parameters
        gender: (T,) str     gender per frame
        trans:  (T, 1, 3)    GT SMPL translation parameters
    meta_seq_keys:      (T, 3)       per-frame NPZ key: subject, action, camera index
    meta_frame_indices: (T,)         per-frame index in NPZ sequence
    image_paths:        (T,)         image file paths (optional, with --add-image)
    keypoints_2d:       (T, 17, 2)   2D keypoints (optional, with --add-image)

Usage:
    # Convert MixSTE predictions for the VARPose HMR mesh-eval pipeline.
    # Run from third_parties/mixste/.
    python prepare_for_mesh_eval_h36m.py \
        --prediction output/mixste_concat_gt_f_81.npz \
        --format msst \
        --pkl data/msst_data_h36m_vp3d_standard.pkl \
        --output output/mixste_concat_gt_f_81_integrated.npz

    # with image paths and 2D keypoints:
    python prepare_for_mesh_eval_h36m.py \
        --prediction output/mixste_concat_gt_f_81.npz \
        --format msst \
        --pkl data/msst_data_h36m_vp3d_standard.pkl \
        --output output/mixste_concat_gt_f_81_integrated.npz \
        --add-image
"""

import argparse
import pickle
import re
import numpy as np
import os

CAM_NAME = ['54138969', '55011271', '58860488', '60457274']

MSST_ACTION_NAMES = [
    'Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning',
    'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking',
    'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether'
]

MSST_STATIC_ACTION_MAP = {
    'S1': {
        ('Directions', 1): 'Directions 1', ('Directions', 2): 'Directions',
        ('Discussion', 1): 'Discussion 1', ('Discussion', 2): 'Discussion',
        ('Eating', 1): 'Eating 2', ('Eating', 2): 'Eating',
        ('Greeting', 1): 'Greeting 1', ('Greeting', 2): 'Greeting',
        ('Phoning', 1): 'Phoning 1', ('Phoning', 2): 'Phoning',
        ('Photo', 1): 'Photo 1', ('Photo', 2): 'Photo',
        ('Posing', 1): 'Posing 1', ('Posing', 2): 'Posing',
        ('Purchases', 1): 'Purchases 1', ('Purchases', 2): 'Purchases',
        ('Sitting', 1): 'Sitting 1', ('Sitting', 2): 'Sitting 2',
        ('SittingDown', 1): 'SittingDown 2', ('SittingDown', 2): 'SittingDown',
        ('Smoking', 1): 'Smoking 1', ('Smoking', 2): 'Smoking',
        ('Waiting', 1): 'Waiting 1', ('Waiting', 2): 'Waiting',
        ('WalkDog', 1): 'WalkDog 1', ('WalkDog', 2): 'WalkDog',
        ('WalkTogether', 1): 'WalkTogether 1', ('WalkTogether', 2): 'WalkTogether',
        ('Walking', 1): 'Walking 1', ('Walking', 2): 'Walking'
    },
    'S5': {
        ('Directions', 1): 'Directions 1', ('Directions', 2): 'Directions 2',
        ('Discussion', 1): 'Discussion 2', ('Discussion', 2): 'Discussion 3',
        ('Eating', 1): 'Eating 1', ('Eating', 2): 'Eating',
        ('Greeting', 1): 'Greeting 1', ('Greeting', 2): 'Greeting 2',
        ('Phoning', 1): 'Phoning 1', ('Phoning', 2): 'Phoning',
        ('Photo', 1): 'Photo', ('Photo', 2): 'Photo 2',
        ('Posing', 1): 'Posing 1', ('Posing', 2): 'Posing',
        ('Purchases', 1): 'Purchases 1', ('Purchases', 2): 'Purchases',
        ('Sitting', 1): 'Sitting 1', ('Sitting', 2): 'Sitting',
        ('SittingDown', 1): 'SittingDown', ('SittingDown', 2): 'SittingDown 1',
        ('Smoking', 1): 'Smoking 1', ('Smoking', 2): 'Smoking',
        ('Waiting', 1): 'Waiting 1', ('Waiting', 2): 'Waiting 2',
        ('WalkDog', 1): 'WalkDog 1', ('WalkDog', 2): 'WalkDog',
        ('WalkTogether', 1): 'WalkTogether 1', ('WalkTogether', 2): 'WalkTogether',
        ('Walking', 1): 'Walking 1', ('Walking', 2): 'Walking'
    },
    'S6': {
        ('Directions', 1): 'Directions 1', ('Directions', 2): 'Directions',
        ('Discussion', 1): 'Discussion 1', ('Discussion', 2): 'Discussion',
        ('Eating', 1): 'Eating 1', ('Eating', 2): 'Eating 2',
        ('Greeting', 1): 'Greeting 1', ('Greeting', 2): 'Greeting',
        ('Phoning', 1): 'Phoning 1', ('Phoning', 2): 'Phoning',
        ('Photo', 1): 'Photo', ('Photo', 2): 'Photo 1',
        ('Posing', 1): 'Posing 2', ('Posing', 2): 'Posing',
        ('Purchases', 1): 'Purchases 1', ('Purchases', 2): 'Purchases',
        ('Sitting', 1): 'Sitting 1', ('Sitting', 2): 'Sitting 2',
        ('SittingDown', 1): 'SittingDown 1', ('SittingDown', 2): 'SittingDown',
        ('Smoking', 1): 'Smoking 1', ('Smoking', 2): 'Smoking',
        ('Waiting', 1): 'Waiting 3', ('Waiting', 2): 'Waiting',
        ('WalkDog', 1): 'WalkDog 1', ('WalkDog', 2): 'WalkDog',
        ('WalkTogether', 1): 'WalkTogether 1', ('WalkTogether', 2): 'WalkTogether',
        ('Walking', 1): 'Walking 1', ('Walking', 2): 'Walking'
    },
    'S7': {
        ('Directions', 1): 'Directions 1', ('Directions', 2): 'Directions',
        ('Discussion', 1): 'Discussion 1', ('Discussion', 2): 'Discussion',
        ('Eating', 1): 'Eating 1', ('Eating', 2): 'Eating',
        ('Greeting', 1): 'Greeting 1', ('Greeting', 2): 'Greeting',
        ('Phoning', 1): 'Phoning 2', ('Phoning', 2): 'Phoning',
        ('Photo', 1): 'Photo', ('Photo', 2): 'Photo 1',
        ('Posing', 1): 'Posing 1', ('Posing', 2): 'Posing',
        ('Purchases', 1): 'Purchases 1', ('Purchases', 2): 'Purchases',
        ('Sitting', 1): 'Sitting 1', ('Sitting', 2): 'Sitting',
        ('SittingDown', 1): 'SittingDown', ('SittingDown', 2): 'SittingDown 1',
        ('Smoking', 1): 'Smoking 1', ('Smoking', 2): 'Smoking',
        ('Waiting', 1): 'Waiting 1', ('Waiting', 2): 'Waiting 2',
        ('WalkDog', 1): 'WalkDog 1', ('WalkDog', 2): 'WalkDog',
        ('WalkTogether', 1): 'WalkTogether 1', ('WalkTogether', 2): 'WalkTogether',
        ('Walking', 1): 'Walking 1', ('Walking', 2): 'Walking 2'
    },
    'S8': {
        ('Directions', 1): 'Directions 1', ('Directions', 2): 'Directions',
        ('Discussion', 1): 'Discussion 1', ('Discussion', 2): 'Discussion',
        ('Eating', 1): 'Eating 1', ('Eating', 2): 'Eating',
        ('Greeting', 1): 'Greeting 1', ('Greeting', 2): 'Greeting',
        ('Phoning', 1): 'Phoning 1', ('Phoning', 2): 'Phoning',
        ('Photo', 1): 'Photo 1', ('Photo', 2): 'Photo',
        ('Posing', 1): 'Posing 1', ('Posing', 2): 'Posing',
        ('Purchases', 1): 'Purchases 1', ('Purchases', 2): 'Purchases',
        ('Sitting', 1): 'Sitting 1', ('Sitting', 2): 'Sitting',
        ('SittingDown', 1): 'SittingDown', ('SittingDown', 2): 'SittingDown 1',
        ('Smoking', 1): 'Smoking 1', ('Smoking', 2): 'Smoking',
        ('Waiting', 1): 'Waiting 1', ('Waiting', 2): 'Waiting',
        ('WalkDog', 1): 'WalkDog 1', ('WalkDog', 2): 'WalkDog',
        ('WalkTogether', 1): 'WalkTogether 1', ('WalkTogether', 2): 'WalkTogether 2',
        ('Walking', 1): 'Walking 1', ('Walking', 2): 'Walking'
    },
    'S9': {
        ('Directions', 1): 'Directions 1', ('Directions', 2): 'Directions',
        ('Discussion', 1): 'Discussion 1', ('Discussion', 2): 'Discussion 2',
        ('Eating', 1): 'Eating 1', ('Eating', 2): 'Eating',
        ('Greeting', 1): 'Greeting 1', ('Greeting', 2): 'Greeting',
        ('Phoning', 1): 'Phoning 1', ('Phoning', 2): 'Phoning',
        ('Photo', 1): 'Photo 1', ('Photo', 2): 'Photo',
        ('Posing', 1): 'Posing 1', ('Posing', 2): 'Posing',
        ('Purchases', 1): 'Purchases 1', ('Purchases', 2): 'Purchases',
        ('Sitting', 1): 'Sitting 1', ('Sitting', 2): 'Sitting',
        ('SittingDown', 1): 'SittingDown', ('SittingDown', 2): 'SittingDown 1',
        ('Smoking', 1): 'Smoking 1', ('Smoking', 2): 'Smoking',
        ('Waiting', 1): 'Waiting 1', ('Waiting', 2): 'Waiting',
        ('WalkDog', 1): 'WalkDog 1', ('WalkDog', 2): 'WalkDog',
        ('WalkTogether', 1): 'WalkTogether 1', ('WalkTogether', 2): 'WalkTogether',
        ('Walking', 1): 'Walking 1', ('Walking', 2): 'Walking'
    },
    'S11': {
        ('Directions', 1): 'Directions 1', ('Directions', 2): 'Directions',
        ('Discussion', 1): 'Discussion 1', ('Discussion', 2): 'Discussion 2',
        ('Eating', 1): 'Eating 1', ('Eating', 2): 'Eating',
        ('Greeting', 1): 'Greeting 2', ('Greeting', 2): 'Greeting',
        ('Phoning', 1): 'Phoning 3', ('Phoning', 2): 'Phoning 2',
        ('Photo', 1): 'Photo 1', ('Photo', 2): 'Photo',
        ('Posing', 1): 'Posing 1', ('Posing', 2): 'Posing',
        ('Purchases', 1): 'Purchases 1', ('Purchases', 2): 'Purchases',
        ('Sitting', 1): 'Sitting 1', ('Sitting', 2): 'Sitting',
        ('SittingDown', 1): 'SittingDown', ('SittingDown', 2): 'SittingDown 1',
        ('Smoking', 1): 'Smoking 2', ('Smoking', 2): 'Smoking',
        ('Waiting', 1): 'Waiting 1', ('Waiting', 2): 'Waiting',
        ('WalkDog', 1): 'WalkDog 1', ('WalkDog', 2): 'WalkDog',
        ('WalkTogether', 1): 'WalkTogether 1', ('WalkTogether', 2): 'WalkTogether',
        ('Walking', 1): 'Walking 1', ('Walking', 2): 'Walking'
    }
}

def action_name_msst_to_npz(subject_idx:str, action_idx:str, subaction_idx:int):
    return MSST_STATIC_ACTION_MAP[subject_idx][(MSST_ACTION_NAMES[action_idx, subaction_idx])]

FRAME_NUM_RE = re.compile(r'(\d+)\.jpg$')


def get_image_path(meta_seq_key, meta_frame_idx, image_dir):
    return os.path.join(image_dir, meta_seq_key[0], meta_seq_key[1], 
                        CAM_NAME[meta_seq_key[2]], f'frame_{meta_frame_idx:06d}.jpg')


def get_keypoints_2d(meta_seq_key, meta_frame_idx, data_2d_npz):
    s, action, cam = meta_seq_key[0], meta_seq_key[1], meta_seq_key[2]
    action_data = None
    if action in data_2d_npz[s]:
        action_data = data_2d_npz[s][action]
    else:
        raise ValueError(f"Cannot find action for {s}/{action}")
    n_frames = action_data[cam].shape[0]
    frame_idx = meta_frame_idx if meta_frame_idx < n_frames else n_frames - 1
    return action_data[cam][frame_idx]


def get_action_name_msst(action_idx, subaction_idx):
    idx = action_idx - 2
    if idx < 0 or idx >= len(MSST_ACTION_NAMES):
        return None
    return f'{MSST_ACTION_NAMES[idx]} {subaction_idx}'


def get_action_name_vp3d(action_idx, subaction_idx, subject):
    s_id = f'S{subject}'
    idx = action_idx - 2
    if idx < 0 or idx >= len(MSST_ACTION_NAMES):
        return None
    action_name = MSST_ACTION_NAMES[idx]
    key = (action_name, subaction_idx)
    if s_id in MSST_STATIC_ACTION_MAP and key in MSST_STATIC_ACTION_MAP[s_id]:
        return MSST_STATIC_ACTION_MAP[s_id][key]
    return None


def main():
    parser = argparse.ArgumentParser(description="Prepare integrated NPZ for H36M mesh evaluation")
    parser.add_argument("--prediction", required=False, default=None, help="Path to prediction NPZ")
    parser.add_argument("--pkl", required=True, help="Path to PKL with SMPL metadata")
    parser.add_argument("--output", required=True, help="Output NPZ path")
    parser.add_argument("--format", default="msst", choices=["msst", "vp3d"],
                        help="NPZ key format: msst or vp3d (default: msst)")
    parser.add_argument("--gt-3d", action="store_true", help='using gt_3d in pkl.')
    parser.add_argument("--add-image", action="store_true", help='add image_paths and keypoints_2d')
    parser.add_argument("--image-dir", type=str, 
                        default='/data/human3.6m/data_from_csdn/h36m/images/',
                        help='Base directory for images (used with --add-image)')
    parser.add_argument("--data-2d", type=str,
                        default="/data/human3.6m/processedByVideoPose3d/data/data_2d_h36m_gt.npz",
                        help='2D keypoints npz file (used with --add-image)')
    args = parser.parse_args()

    pred_dict = None
    if not args.gt_3d:
        if args.prediction is None:
            raise ValueError("--prediction is required when not using --gt-3d")
        print("Loading prediction NPZ...")
        pred_file = np.load(args.prediction, allow_pickle=True)
        pred_dict = pred_file['predictions_dict'].item()

    print("Loading PKL (this may take a while)...")
    with open(args.pkl, 'rb') as f:
        pkl_data = pickle.load(f)
    test_coords = pkl_data['test']['coords']
    T = len(test_coords)
    print(f"PKL test frames: {T}, format: {args.format}")

    all_pred, all_theta, all_beta, all_gender, all_trans = [], [], [], [], []
    meta_seq_keys, meta_frame_indices = [], []
    image_paths, keypoints_2d = [], []

    if args.add_image:
        data_2d_npz = np.load(args.data_2d, allow_pickle=True)['positions_2d'].item()
        print(f"Loaded 2D keypoints from {args.data_2d}")

    skipped = 0
    for i, entry in enumerate(test_coords):
        meta = entry['metadata']
        subject = meta['subject']
        action_idx = meta['action_idx']
        subaction_idx = meta['subaction_idx']
        cam_idx = meta['cam_idx']

        if args.format == "msst":
            action_name = get_action_name_msst(action_idx, subaction_idx)
            action_name_vp3d = get_action_name_vp3d(action_idx, subaction_idx, subject)
        else:
            action_name = get_action_name_vp3d(action_idx, subaction_idx, subject)
            action_name_vp3d = action_name

        if action_name is None:
            skipped += 1
            continue
        

        npz_key = (f'S{subject}', action_name, cam_idx - 1)
        npz_key_vp3d = (f'S{subject}', action_name_vp3d, cam_idx - 1)
        match = FRAME_NUM_RE.search(meta['img_name'])
        frame_num = int(match.group(1))
        frame_idx = frame_num - 1
        if args.gt_3d:
            all_pred.append(entry["3d"][-1]/1000)
        else:
            if npz_key not in pred_dict:
                skipped += 1
                continue

            seq = pred_dict[npz_key]

            if frame_idx >= seq.shape[0]:
                skipped += 1
                continue

            all_pred.append(seq[frame_idx])
        sp = meta['smpl_param']
        all_theta.append(np.array(sp['pose'], dtype=np.float64))
        all_beta.append(np.array(sp['shape'], dtype=np.float64))
        all_gender.append(sp['gender'])
        all_trans.append(np.array(sp['trans'], dtype=np.float64))
        meta_seq_keys.append(npz_key)
        meta_frame_indices.append(frame_idx)

        if args.add_image:
            image_paths.append(get_image_path(npz_key_vp3d, frame_idx, args.image_dir))
            keypoints_2d.append(get_keypoints_2d(npz_key_vp3d, frame_idx, data_2d_npz))

        if (i + 1) % 2000 == 0:
            print(f"  Processed {i + 1}/{T} frames...")

    prediction = np.stack(all_pred)
    theta = np.stack(all_theta)
    beta = np.stack(all_beta)
    gender = np.array(all_gender)
    trans = np.stack(all_trans)
    meta_seq_keys = np.array(meta_seq_keys, dtype=object)
    meta_frame_indices = np.array(meta_frame_indices, dtype=np.int64)

    T_out = prediction.shape[0]
    print(f"\nOutput frames: {T_out} (skipped {skipped})")
    print(f"prediction: {prediction.shape}, dtype={prediction.dtype}")
    print(f"theta:      {theta.shape}, dtype={theta.dtype}")
    print(f"beta:       {beta.shape}, dtype={beta.dtype}")
    print(f"gender:     {gender.shape}, dtype={gender.dtype}, unique={np.unique(gender)}")
    print(f"trans:       {trans.shape}, dtype={trans.dtype}")

    smpl_param = {'theta': theta, 'beta': beta, 'gender': gender, 'trans': trans}

    output_dict = {
        'prediction': prediction,
        'smpl_param': smpl_param,
        'meta_seq_keys': meta_seq_keys,
        'meta_frame_indices': meta_frame_indices,
    }

    if args.add_image:
        image_paths_arr = np.array(image_paths, dtype=object)
        keypoints_2d_arr = np.stack(keypoints_2d, axis=0)
        output_dict['image_paths'] = image_paths_arr
        output_dict['keypoints_2d'] = keypoints_2d_arr
        print(f"image_paths: {image_paths_arr.shape}, keypoints_2d: {keypoints_2d_arr.shape}")

    np.savez(args.output, **output_dict)
    print(f"Saved to {args.output}")

    v = np.load(args.output, allow_pickle=True)
    assert v['prediction'].shape == (T_out, 17, 3)
    sp_out = v['smpl_param'].item()
    assert sp_out['theta'].shape == (T_out, 72)
    assert sp_out['beta'].shape == (T_out, 10)
    assert sp_out['gender'].shape == (T_out,)
    assert sp_out['trans'].shape == (T_out, 1, 3)
    assert v['meta_seq_keys'].shape[0] == T_out
    assert v['meta_frame_indices'].shape == (T_out,)
    print("Verification passed.")


if __name__ == "__main__":
    main()
