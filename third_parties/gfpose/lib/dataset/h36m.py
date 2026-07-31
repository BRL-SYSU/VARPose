import numpy as np
import os, sys
import pickle
from prettytable import PrettyTable
from collections import defaultdict
import heapq

from lib.utils.transforms import image_to_camera_frame, align_to_gt

# from multiprocessing import Pool


def flip_data(data):
    """
    horizontal flip
        data: [N, 17*k] or [N, 17, k], i.e. [x, y], [x, y, confidence] or [x, y, z]
    Return
        result: [2N, 17*k] or [2N, 17, k]
    """
    left_joints = [4, 5, 6, 11, 12, 13]
    right_joints = [1, 2, 3, 14, 15, 16]

    flipped_data = data.copy().reshape((len(data), 17, -1))
    flipped_data[:, :, 0] *= -1  # flip x of all joints
    flipped_data[:, left_joints+right_joints] = flipped_data[:, right_joints+left_joints]
    flipped_data = flipped_data.reshape(data.shape)

    result = np.concatenate((data, flipped_data), axis=0)

    return result

def unflip_data(data):
    """
    Average original data and flipped data
        data: [2N, 17*3]
    Return
        result: [N, 17*3]
    """
    left_joints = [4, 5, 6, 11, 12, 13]
    right_joints = [1, 2, 3, 14, 15, 16]

    data = data.copy().reshape((2, -1, 17, 3))
    data[1, :, :, 0] *= -1  # flip x of all joints
    data[1, :, left_joints+right_joints] = data[1, :, right_joints+left_joints]
    data = np.mean(data, axis=0)
    data = data.reshape((-1, 17*3))

    return data

def denormalize_data(data, which='scale'):
    """
    data: [B, j, 3]
    Return: [B, j, 3]
    """
    res_w, res_h = 1000, 1000
    assert data.ndim >= 3
    if which == 'scale':
        data = data.copy()
        data[..., :2] = (data[..., :2] + [1, res_h / res_w]) * res_w / 2
        data[..., 2:] = data[..., 2:] * res_w / 2
    else:
        assert 0
    return data

def normalize_data(data):
    """
    data: [B, j, 3]
    Return: [B, j, 3]
    """
    res_w, res_h = 1000, 1000
    assert data.ndim >= 3
    data = data.copy()
    data[..., :2] = data[..., :2] / res_w * 2 - [1, res_h / res_w]
    data[..., 2:] = data[..., 2:] / res_w * 2
    return data

def worker(args):
    multi_pred, box, camera_param, root_depth, gt, protocol2 = args
    multi_results = []
    for pred in multi_pred:
        pred = image_to_camera_frame(pose3d_image_frame=pred,
            box=box,
            camera=camera_param, rootIdx=0,
            root_depth=root_depth)
        if protocol2:
            pred = align_to_gt(pose=pred, pose_gt=gt)
        error_per_joint = np.sqrt(np.square(pred-gt).sum(axis=1))  # [17]
        multi_results.append(np.mean(error_per_joint))  # scala
    return np.amin(multi_results)  # min error among multi-hypothesis


class H36MDataset3D:
    def __init__(self, root_path, subset='train', 
        gt2d=True, read_confidence=False, sample_interval=None, rep=1, 
        flip=False, cond_3d_prob=0, dataset_name='h36m', use_dense=True, dataset_path="",
        detector_dataset_path=""):
        
        self.gt_trainset = None
        self.gt_testset = None
        self.dt_dataset = None
        self.root_path = root_path
        self.subset = subset
        self.gt2d = gt2d
        self.read_confidence = read_confidence
        self.sample_interval = sample_interval
        self.flip = flip
        self.dataset_name = dataset_name
        self.use_dense = use_dense
        self.dataset_path = dataset_path
        self.detector_dataset_path = detector_dataset_path
        self.h36m_14_eval_joints = [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16]

        self.db_2d, self.db_3d, self.gt_dataset = self.read_data()

        if self.sample_interval:
            self._sample(sample_interval)

        self.rep = rep
        if self.rep > 1:
            print(f'stack dataset {self.rep} times for multi-sample eval')

        self.real_data_len = len(self.db_2d)

        self.left_joints = [4, 5, 6, 11, 12, 13]
        self.right_joints = [1, 2, 3, 14, 15, 16]
        symmetry_48 = np.array([(0, 36), (1, 7), (2, 47), (3, 4), (6, 19), (9, 26), (11, 14), (12, 33), (13, 32), (15, 22), (17, 24), (18, 38), (25, 42), (27, 46), (29, 37), (30, 43), (41, 44)])+17
        symmetry_96 = np.array([(0, 72), (2, 14), (4, 95), (6, 9), (8, 68), (12, 38), (13, 39), (16, 80), (18, 52), (20, 41), (21, 30), (22, 28), (24, 66), (26, 64), (31, 45), (32, 59), (35, 49), (37, 76), (42, 87), (50, 84), (51, 85), (54, 93), (56, 77), (57, 75), (58, 74), (60, 86), (82, 89)])+17+48
        symmetry_dense = np.concatenate([symmetry_48, symmetry_96], axis=0)
        self.kps_dense_left = list(symmetry_dense[:, 0])
        self.kps_dense_right = list(symmetry_dense[:, 1])
        pseudo_joints_idx_48 = np.array([23, 35, 39])+17
        pseudo_joints_idx_96 = np.array([3, 7, 11, 15, 19, 23, 27, 46, 47, 70, 71, 78, 79])+17+48
        self.pseudo_joints_idx = np.concatenate([pseudo_joints_idx_48, pseudo_joints_idx_96], axis=0)

        self.cond_3d_prob = cond_3d_prob

    def __getitem__(self, idx):
        """
        Return: [17, 2], [17, 3] for data and labels
        """
        data_2d = self.db_2d[idx % self.real_data_len]
        data_3d = self.db_3d[idx % self.real_data_len]


        # always return [17, 3] for data_2d
        n_joints = len(data_2d)
        data_2d = np.concatenate(
            (data_2d, np.zeros((n_joints, 1), dtype=np.float32)),
            axis=-1,
        )  # [17, 3]

        # return gt3d in some prob while training
        if self.cond_3d_prob and self.subset == 'train':
            if np.random.rand(1,)[0] < self.cond_3d_prob:
                # return 3d
                data_2d = data_3d

        # only random flip during training
        if self.flip and self.subset == 'train':
            data_2d = self._random_flip(data_2d)
            data_3d = self._random_flip(data_3d)

        expected_2d_shape = (161, 3) if self.use_dense else (17, 3)
        if data_2d.shape != expected_2d_shape:
            print(f"Warning: data_2d shape {data_2d.shape} != expected {expected_2d_shape}")

        return data_2d, data_3d

    def __len__(self,):
        # assert len(self.db_2d) == len(self.db_3d)
        return len(self.db_2d) * self.rep

    def _random_flip(self, data, p=0.5):
        """
        Flip with prob p
        data: [17, 2] or [17, 3]
        """
        if np.random.rand(1,)[0] < p:
            data = data.copy()
            data[:, 0] *= -1  # flip x of all joints
            if data.shape[-2]==17:
                data[self.left_joints+self.right_joints] = data[self.right_joints+self.left_joints]
            else:
                left_joints = self.left_joints + self.kps_dense_left
                right_joints = self.right_joints + self.kps_dense_right
                data[left_joints+right_joints] = data[right_joints+left_joints]

        return data

    def add_noise(self, pose2d, std=5, noise_type='gaussian'):
        """
        pose2d: [B, j, 2]
        """
        if noise_type == 'gaussian':
            noise = std * np.random.randn(*pose2d.shape).astype(np.float32)
            pose2d = pose2d + noise
        elif noise_type == 'uniform':
            # a range of [-0.5std, 0.5std]
            noise = std * (np.random.rand(*pose2d.shape).astype(np.float32) - 0.5)
            pose2d = pose2d + noise
        else:
            raise NotImplementedError
        return pose2d

    def _sample(self, sample_interval):
        print(f'Class H36MDataset({self.subset}): sample dataset every {sample_interval} frame')
        self.db_2d = self.db_2d[::sample_interval]
        self.db_3d = self.db_3d[::sample_interval]
        self.gt_dataset = self.gt_dataset[::sample_interval]

    def read_data(self):
        # read 3d labels
        file_name = self.dataset_path
        
        print('loading %s dataset: %s (use_dense=%s)' % (self.dataset_name, file_name, self.use_dense))
        file_path = os.path.join(self.root_path, file_name)
        with open(file_path, 'rb') as f:
            gt_dataset = pickle.load(f)

        if isinstance(gt_dataset, dict):
        
            if self.subset in gt_dataset:
                gt_dataset = gt_dataset[self.subset]
            else:
                gt_dataset = gt_dataset.get('test', list(gt_dataset.values())[0])
        else:
            # H36M
            gt_dataset = gt_dataset

        # normalize
        res_w, res_h = 1000, 1000
        labels_3d = np.empty((len(gt_dataset), 17, 3), dtype=np.float32)  # [N, 17, 3]
        
        # map to [-1, 1]
        for idx, item in enumerate(gt_dataset):
            labels_3d[idx] = item['joint_3d_image']

        labels_3d[..., :2] = labels_3d[..., :2] / res_w * 2 - [1, res_h / res_w]
        labels_3d[..., 2:] = labels_3d[..., 2:] / res_w * 2

        # read 2d
        if self.gt2d:
            data_2d = labels_3d[..., :2].copy()  # [N, 17, 2]
            
            if self.use_dense:
                dense_2d = np.zeros((len(gt_dataset), 144, 2), dtype=np.float32)
                for idx, item in enumerate(gt_dataset):
                    dense_2d[idx] = item['dense_2d']
                dense_2d = dense_2d / res_w * 2 - [1, res_h / res_w]
                data_2d = np.concatenate([data_2d, dense_2d], axis=-2)  # [N, 161, 2]
            
            if self.read_confidence:
                if self.use_dense:
                    confidence = np.ones((len(data_2d), 161, 1), dtype=np.float32)
                    data_2d = np.concatenate((data_2d, confidence), axis=-1)  # [N, 161, 3]
                else:
                    data_2d = np.concatenate((data_2d, np.ones((len(data_2d), 17, 1))), axis=-1)  # [N, 17, 3]
        else:
            if not self.detector_dataset_path:
                raise ValueError("detector_dataset_path is required when gt2d=False. "
                    "Please provide a path to the detector 2D output file.")
            file_path = os.path.join(self.root_path, self.detector_dataset_path)
            print('loading dt_2d %s' % file_path)
            with open(file_path, 'rb') as f:
                dt_dataset = pickle.load(f)

            data_2d = dt_dataset[self.subset]['joint3d_image'][:, :, :2].copy()  # [N, 17, 2]
            
            if self.use_dense:
                dense_2d = dt_dataset[self.subset]['dense_2d'][:, :, :2].copy()
                data_2d = np.concatenate([data_2d, dense_2d], axis=-2)  # [N, 161, 2]
            
            data_2d = data_2d / res_w * 2 - [1, res_h / res_w]

            if self.read_confidence:
                dt_confidence = dt_dataset[self.subset]['confidence'].copy()  # [N, 17, 1]
                if self.use_dense:
                    dense_confidence = np.ones((len(dt_confidence), 144, 1), dtype=np.float32)
                    dt_confidence = np.concatenate([dt_confidence, dense_confidence], axis=1)  # [N, 161, 1]
                data_2d = np.concatenate((data_2d, dt_confidence), axis=-1)
        
        data_2d = data_2d.astype(np.float32)

        expected_2d_shape = (len(gt_dataset), 161, 3) if self.use_dense else (len(gt_dataset), 17, 3)
        if data_2d.shape[:2] != expected_2d_shape[:2]:
            print(f"Warning: data_2d shape {data_2d.shape} != expected {expected_2d_shape}")

        return data_2d, labels_3d, gt_dataset

    def eval(self, preds, protocol2=False, print_verbose=False, sample_interval=None):
        """
        Eval action-wise MPJPE
        preds: [N, j, 3]
        sample_interval: eval every 
        Return: MPJPE, scala
        """
        print('eval...')

        # read testset
        assert self.subset == 'test' and getattr(self, 'gt_dataset', False), \
            f"eval() requires gt_dataset (subset={self.subset}, " \
            f"gt_dataset={'loaded' if getattr(self, 'gt_dataset', None) else 'missing'})"
        dataitem_gt = self.gt_dataset

        # read preds
        # result_path = os.path.join(ROOT_PATH, 'experiment', test_name, 'result_%s.pkl' % mode)
        # with open(result_path, 'rb') as f:
        #     preds = pickle.load(f)['result']  # [N, 17, 3]
        # preds = np.reshape(preds, (-1, 17, 3))

        assert len(preds) == len(dataitem_gt)

        if sample_interval is not None:
            preds = preds[::sample_interval]

        results = []
        for idx, pred in enumerate(preds):
            pred = image_to_camera_frame(pose3d_image_frame=pred, box=dataitem_gt[idx]['box'],
                camera=dataitem_gt[idx]['camera_param'], rootIdx=0,
                root_depth=dataitem_gt[idx]['root_depth'])
            gt = dataitem_gt[idx]['joint_3d_camera']
            pred = pred - pred[..., 0:1, :]
            gt = gt - gt[..., 0:1, :]
            
            if protocol2:
                pred = align_to_gt(pose=pred, pose_gt=gt)

            error_per_joint = np.sqrt(np.square(pred-gt).sum(axis=1))  # [17]
            results.append(error_per_joint)
            # if idx % 10000 == 0:
            #     print('step:%d' % idx + '-' * 20)
            #     print(np.mean(error_per_joint))
        results = np.array(results)  # [N ,17]

        # action-wise MPJPE
        final_result = []
        action_index_dict = {}
        for i in range(2, 17):
            action_index_dict[i] = []
        for idx, dataitem in enumerate(dataitem_gt):
            action_index_dict[dataitem['action']].append(idx)
        for i in range(2, 17):
            if len(action_index_dict[i]) > 0:
                final_result.append(np.mean(results[action_index_dict[i]]))
            else:
                final_result.append(np.nan)
        # Calculate average excluding NaN values and append
        error = np.nanmean(np.array(final_result))
        final_result.append(error)

        # print error
        if print_verbose:
            table = PrettyTable()
            table.field_names = ['H36M'] + [i for i in range(2, 17)] + ['avg']
            table.add_row(['p2' if protocol2 else 'p1'] + ['%.2f' % d for d in final_result])
            print(table)

        return error

    def eval_multi(self, preds, protocol2=False, print_verbose=False, sample_interval=None):
        """
        Eval action-wise MPJPE
        preds: [N, m, j, 3], N:len of dataset, m: multi-hypothesis number
        sample_interval: eval every 
        Return: MPJPE, scala
        """
        is_3dpw = '3dpw' in self.dataset_name.lower()
        eval_indices = self.h36m_14_eval_joints if is_3dpw else list(range(17))
        metric_name = "PA-MPJPE" if protocol2 else "MPJPE"

        mode_str = "14j" if is_3dpw else "17j"
        print(f'eval multi-hypothesis (Dataset: {self.dataset_name}, Mode: {mode_str})...')

        # read testset
        assert self.subset == 'test' and getattr(self, 'gt_dataset', False), \
            f"eval_multi() requires gt_dataset (subset={self.subset}, " \
            f"gt_dataset={'loaded' if getattr(self, 'gt_dataset', None) else 'missing'})"
        dataitem_gt = self.gt_dataset

        assert len(preds) == len(dataitem_gt)

        if sample_interval is not None:
            preds = preds[::sample_interval]

        results = []
        multi_preds_cam = []
        for idx, multi_pred in enumerate(preds):
            multi_results = []
            pred_store = []
            for pred in multi_pred:
                pred = image_to_camera_frame(pose3d_image_frame=pred, box=dataitem_gt[idx]['box'],
                    camera=dataitem_gt[idx]['camera_param'], rootIdx=0,
                    root_depth=dataitem_gt[idx]['root_depth'])
                gt = dataitem_gt[idx]['joint_3d_camera']
                pred = pred - pred[..., 0:1, :]
                gt = gt - gt[..., 0:1, :]
                pred_store.append(pred)
                
                if protocol2:
                    pred = align_to_gt(pose=pred, pose_gt=gt)

                error_per_joint = np.sqrt(np.square(pred-gt).sum(axis=1))  # [17]
                error_per_joint = error_per_joint[eval_indices]
                multi_results.append(np.mean(error_per_joint))  # scala
            results.append(np.amin(multi_results))  # min error among multi-hypothesis
            multi_preds_cam.append(pred_store)  # [M, j, 3]
        results = np.array(results)  # [N]
        multi_preds_cam = np.array(multi_preds_cam)  # [N, M, j, 3]

        # diversity in std, expcet root joints
        multi_preds_cam_eval = multi_preds_cam - multi_preds_cam[:, :, [0], :]
        multi_preds_cam_eval = multi_preds_cam_eval[:, :, 1:, :]  # [N, M, j-1, 3]
        print(f'std: x{multi_preds_cam_eval[..., 0].std(axis=1).mean()}, \
            y{multi_preds_cam_eval[..., 1].std(axis=1).mean()}, z{multi_preds_cam_eval[..., 2].std(axis=1).mean()}')

        if is_3dpw:
            vid_errors = defaultdict(list)
            for idx, dataitem in enumerate(dataitem_gt):
                v_name = dataitem.get('vid_name', 'unknown_seq')
                vid_errors[v_name].append(results[idx])
            
            sorted_vids = sorted(vid_errors.keys())
            seq_means = [np.mean(vid_errors[v]) for v in sorted_vids]
            total_avg = np.mean(seq_means)

            if print_verbose:
                table = PrettyTable()
                table.field_names = ["3DPW Sequence Name", f"{metric_name} (14j)" if is_3dpw else f"{metric_name} (17j)"]
                table.align["3DPW Sequence Name"] = "l"
                
                for v_name, m_err in zip(sorted_vids, seq_means):
                    table.add_row([v_name, f"{m_err:.2f}"])
                
                table.add_row(["====================", "======="])
                table.add_row(["OVERALL AVERAGE", f"{total_avg:.2f}"])
                print(table)
            
            return total_avg

        else:
            # action-wise MPJPE
            final_result = []
            action_index_dict = {}
            for i in range(2, 17):
                action_index_dict[i] = []
            for idx, dataitem in enumerate(dataitem_gt):
                action_index_dict[dataitem['action']].append(idx)
            for i in range(2, 17):
                if len(action_index_dict[i]) > 0:
                    final_result.append(np.mean(results[action_index_dict[i]]))
                else:
                    final_result.append(np.nan)
            # Calculate average excluding NaN values and append
            error = np.nanmean(np.array(final_result))
            final_result.append(error)

            # print error
            if print_verbose:
                table = PrettyTable()
                table.field_names = ['H36M'] + [i for i in range(2, 17)] + ['avg']
                table.add_row(['p2' if protocol2 else 'p1'] + ['%.2f' % d for d in final_result])
                print(table)

            return error

    @staticmethod
    def get_skeleton():
        return [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], 
        [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], 
        [8, 14], [14, 15], [15, 16]]
