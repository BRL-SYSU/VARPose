import os
import numpy as np
import functools
import pprint
import sys
import traceback
import argparse
from pathlib import Path

from absl import app
from absl import flags
from absl.flags import argparse_flags
from ml_collections.config_flags import config_flags

# torch related
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

try:
    from tensorboardX import SummaryWriter
except ImportError as e:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as e:
        print('Tensorboard is not Installed')

from lib.utils.generic import create_logger

from lib.algorithms.advanced.model import ScoreModelFC_Adv
from lib.algorithms.advanced import losses, sde_lib, sampling
from lib.algorithms.ema import ExponentialMovingAverage

from lib.dataset.h36m import H36MDataset3D, denormalize_data


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])

# global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_JOINTS = 17
JOINT_DIM = 3
HIDDEN_DIM = 1024
EMBED_DIM = 512
CONDITION_DIM = 3
# BATCH_SIZE = 100000
# TEST_BATCH_SIZE = 10000
N_EPOCHES = 100
EVAL_FREQ = 5  # 20


def parse_args(argv):
    parser = argparse_flags.ArgumentParser(description='train score model')

    # parser.add_argument('--prior-t0', type=float, default=0.5)
    # parser.add_argument('--test-num', type=int, default=20)
    # parser.add_argument('--sample-steps', type=int, default=2000)
    parser.add_argument('--restore-dir', type=str)
    parser.add_argument('--gt', action='store_true', 
        default=False, help='use gt2d as condition')
    parser.add_argument('--sample', type=int, help='sample trainset to reduce data')
    parser.add_argument('--flip', default=False, action='store_true', help='random flip pose during training')
    parser.add_argument('--fusion-epochs-ratio', type=float, default=0.0,
        help='ratio of epochs to train FusionPose only at the beginning (0=disabled)')
    parser.add_argument('--finetune-from', type=str,
        help='path to pretrained checkpoint for fine-tuning (only loads model weights, not training state)')
    parser.add_argument('--backbone-lr-scale', type=float, default=0.1,
        help='learning rate scale for backbone (non-fusion) parameters')

    parser.add_argument('--smoke-test', action='store_true', 
        default=False, help='run smoke test with minimal data and epochs')
    parser.add_argument('--smoke-epochs', type=int, default=3,
        help='number of epochs for smoke test')
    parser.add_argument('--smoke-batches', type=int, default=10,
        help='number of batches per epoch for smoke test')

    # optional
    parser.add_argument('--name', type=str, default='', help='name of checkpoint folder')
    parser.add_argument('--train-dataset-path', type=str, default='data/h36m/h36m_train.pkl',
        help='path to training dataset file')
    parser.add_argument('--test-dataset-path', type=str, default='data/h36m/h36m_test.pkl',
        help='path to test dataset file')
    parser.add_argument('--detector-dataset-path', type=str, default='',
        help='path to detector 2D output file (required when --gt is not set)')

    args = parser.parse_args(argv[1:])

    return args


def get_dataloader(subset='train', sample_interval=None, gt2d=False, flip=False, cond_3d_prob=0,
                   use_dense=True, dataset_path='data/h36m/h36m_train.pkl', detector_dataset_path=''):
    dataset = H36MDataset3D(Path('data', 'h36m'), 
        subset, gt2d=gt2d,
        read_confidence=False, sample_interval=sample_interval,
        flip=flip,
        cond_3d_prob=cond_3d_prob,
        use_dense=use_dense,
        dataset_path=dataset_path,
        detector_dataset_path=detector_dataset_path)
    print(f'H36M 3D {subset} dataset with 3D conditional prob: {cond_3d_prob}')
    
    if subset == 'train':
        # train_labels = torch.FloatTensor(train_labels).reshape((-1, 17, 3)) # [N, 17, 3]
        dataloader = DataLoader(dataset, 
            batch_size=FLAGS.config.training.batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True)
    else:
        # test_labels = torch.FloatTensor(test_labels).reshape((-1, 17, 3)) # [N, 17, 3]
        dataloader = DataLoader(dataset, 
            batch_size=FLAGS.config.eval.batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True)

    return dataloader, dataset


def main(args):
    global N_EPOCHES, EVAL_FREQ
    # args = parse_args()
    config = FLAGS.config

    if args.smoke_test:
        logger, final_output_dir, tb_log_dir = create_logger(
            config, 'smoke_test', folder_name=args.name or 'smoke')
        logger.info("="*50)
        logger.info("SMOKE TEST MODE ENABLED")
        logger.info(f"Epochs: {args.smoke_epochs}, Batches per epoch: {args.smoke_batches}")
        logger.info("="*50)
        N_EPOCHES = args.smoke_epochs
        EVAL_FREQ = 1
    else:
        logger, final_output_dir, tb_log_dir = create_logger(
            config, 'train', folder_name=args.name)
        N_EPOCHES = N_EPOCHES
        EVAL_FREQ = EVAL_FREQ

    logger, final_output_dir, tb_log_dir = create_logger(
            config, 'train', folder_name=args.name)
    logger.info(pprint.pformat(config))
    logger.info(pprint.pformat(args))
    writer = SummaryWriter(tb_log_dir)

    start_epoch = 0  # Start from epoch 0 by default for a fresh training run.

    ''' setup datasets, dataloaders'''
    if args.gt:
        logger.info('use gt data as condition')
    else:
        logger.info('use dt data as condition')
    if args.sample:
        logger.info(f'sample trainset every {args.sample} frame')

    use_dense = config.model.num_dense_joints > 0
    logger.info(f'use_dense={use_dense} (from config.model.num_dense_joints={config.model.num_dense_joints})')
    logger.info(f'train_dataset_path={args.train_dataset_path}')
    logger.info(f'test_dataset_path={args.test_dataset_path}')

    train_loader, train_dataset = get_dataloader('train', args.sample, args.gt, flip=args.flip,
        cond_3d_prob=config.training.cond_3d_prob, use_dense=use_dense, dataset_path=args.train_dataset_path,
        detector_dataset_path=args.detector_dataset_path)
    test_loader, test_dataset = get_dataloader('test', 640, args.gt, flip=False,
        cond_3d_prob=0, use_dense=use_dense, dataset_path=args.test_dataset_path,
        detector_dataset_path=args.detector_dataset_path)  # always sample testset to save time
    logger.info(f'total train samples: {len(train_dataset.db_3d)}')
    logger.info(f'total test samples: {len(test_dataset.db_3d)}')

    ''' setup score networks '''
    # sigma = 25.0  # @param {'type':'number'}
    # marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    # diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)
    model = ScoreModelFC_Adv(
        config,
        n_joints=N_JOINTS,
        joint_dim=JOINT_DIM,
        hidden_dim=HIDDEN_DIM,
        embed_dim=EMBED_DIM,
        cond_dim=CONDITION_DIM,
        # n_blocks=1,
    )
    model.to(device)

    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)

    # Calculate fusion-only epochs
    fusion_only_epochs = int(N_EPOCHES * args.fusion_epochs_ratio)
    
    # Create optimizers for progressive training
    optimizer_all = losses.get_optimizer(config, model.parameters())
    optimizer_fusion = None
    
    if fusion_only_epochs > 0 and hasattr(model, 'fusion_pose'):
        logger.info(f'Progressive training: FusionPose only for first {fusion_only_epochs} epochs')
        optimizer_fusion = losses.get_optimizer(config, model.fusion_pose.parameters())
        # Initially freeze non-fusion parameters
        for name, param in model.named_parameters():
            if 'fusion_pose' not in name:
                param.requires_grad = False
        current_optimizer = optimizer_fusion
    else:
        logger.info('Standard training: full model from the start')
        current_optimizer = optimizer_all
    
    # patience is the number of eval times
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.2)
    # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,120], gamma=0.1)

    state = dict(optimizer=current_optimizer, model=model, ema=ema, step=0)  # based on iteration instead of epochs

    # Fine-tuning mode: load pretrained weights only (no training state)
    if args.finetune_from and os.path.exists(args.finetune_from):
        logger.info(f'=> Fine-tuning from: {args.finetune_from}')
        checkpoint = torch.load(args.finetune_from, map_location=device, weights_only=False)
        # strict=False allows partial parameter loading (e.g., new FusionPose module)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logger.info("=> Loaded pretrained weights (some parameters may be missing)")

        # Reinitialize EMA to match the current model structure.
        ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
        state['ema'] = ema
        logger.info("=> Reinitialized EMA for fine-tuning")

        start_epoch = 0  # Start from epoch 0 in fine-tuning mode
        if args.restore_dir:
            logger.warning("Both --finetune-from and --restore-dir specified. Using fine-tune mode only.")
        # Skip restore logic in fine-tuning mode
    elif args.restore_dir and os.path.exists(args.restore_dir):
        # Standard resume mode: restore training state
        ckpt_path = os.path.join(args.restore_dir, 'checkpoint.pth')
        logger.info(f'=> loading checkpoint: {ckpt_path}')

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        ema.load_state_dict(checkpoint['ema'])
        state['step'] = checkpoint['step']

        # Restore optimizer state based on current training phase
        if fusion_only_epochs > 0 and hasattr(model, 'fusion_pose'):
            if start_epoch < fusion_only_epochs:
                # Resuming in fusion-only phase
                if optimizer_fusion is not None:
                    optimizer_fusion.load_state_dict(checkpoint['optimizer_state_dict'])
                    current_optimizer = optimizer_fusion
                    # Ensure parameters are frozen correctly
                    for name, param in model.named_parameters():
                        if 'fusion_pose' not in name:
                            param.requires_grad = False
                    logger.info(f"Resumed in fusion-only phase (epoch {start_epoch})")
            else:
                # Resuming in full model phase
                optimizer_all.load_state_dict(checkpoint['optimizer_state_dict'])
                current_optimizer = optimizer_all
                # Ensure all parameters are trainable
                for param in model.parameters():
                    param.requires_grad = True
                logger.info(f"Resumed in full model phase (epoch {start_epoch})")
        else:
            # No progressive training
            optimizer_all.load_state_dict(checkpoint['optimizer_state_dict'])
            current_optimizer = optimizer_all

        logger.info(f"=> loaded checkpoint '{ckpt_path}' (epoch {start_epoch})")

    # Identity func
    scaler = lambda x: x
    inverse_scaler = lambda x: x

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                       reduce_mean=False, continuous=continuous,
                                       likelihood_weighting=likelihood_weighting)

    sampling_shape = (config.eval.batch_size, N_JOINTS, JOINT_DIM)
    config.sampling.probability_flow = True
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

    # num_train_steps = config.training.n_iters

    best_error = 1e5
    try:
        ''' training loop '''
        # WARNING!!! This code assumes all poses are normed into [-1, 1]
        for epoch in range(start_epoch, N_EPOCHES):
            # Check if we need to switch from fusion-only to full model training
            if epoch == fusion_only_epochs and fusion_only_epochs > 0:
                logger.info(f'Switching to full model training at epoch {epoch}')
                # Unfreeze all parameters
                for param in model.parameters():
                    param.requires_grad = True
                # Switch optimizer
                current_optimizer = optimizer_all
                state['optimizer'] = current_optimizer
                logger.info('Reinitializing EMA for full model parameters')
                ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
                state['ema'] = ema
                logger.info('All parameters are now trainable')
            
            model.train()

            max_batches = args.smoke_batches if args.smoke_test else len(train_loader)
            for idx, (data_2d, labels_3d) in enumerate(train_loader):
                if args.smoke_test and idx >= max_batches:
                    break

                labels_3d = labels_3d.to(device, non_blocking=True) * config.training.data_scale
                data_2d = data_2d.to(device, non_blocking=True) * config.training.data_scale

                cur_loss = train_step_fn(state, batch=labels_3d, condition=data_2d)

                writer.add_scalar('train_loss', cur_loss.item(), idx + epoch * len(train_loader))

            logger.info(
                f'EPOCH: [{epoch}/{N_EPOCHES}, {epoch/N_EPOCHES*100:.2f}%][{idx}/{max_batches}],\t'
                f'Loss: {cur_loss.item()}'
            )

            ''' eval '''
            if epoch % EVAL_FREQ == 0:
                # sampling process
                model.eval()
                with torch.no_grad():
                    all_results = []

                    max_test_batches = 2 if args.smoke_test else len(test_loader)

                    for idx, (data_2d, labels_3d) in enumerate(test_loader):
                        if args.smoke_test and idx >= max_test_batches:
                            break
                        
                        labels_3d = labels_3d.to(device, non_blocking=True)
                        data_2d = data_2d.to(device, non_blocking=True) * config.training.data_scale

                        # Generate and save samples
                        ema.store(model.parameters())
                        ema.copy_to(model.parameters())
                        trajs, results = sampling_fn(
                            model,
                            condition=data_2d
                        )  # [b ,j ,3]
                        ema.restore(model.parameters())

                        # # trajs: [t, b, j, 3], i.e., the pose-trajs
                        # # results: [b, j, 3], i.e., the end pose of each traj
                        results = results / config.training.data_scale
                        all_results.append(results)

                all_results = np.concatenate(all_results, axis=0)  # [N, j, 3]
                all_results = denormalize_data(all_results)  # [N, j, 3]
                mpjpe = test_dataset.eval(all_results, print_verbose=False)  # scala

                logger.info(f'TEST: [{epoch}/{N_EPOCHES}]')
                logger.info(f'{config.training.sde} {config.sampling.method} sampler: {config.sampling.predictor} {config.sampling.corrector}')
                logger.info(f'TEST:  MPJPE: {mpjpe:.2f}')
                writer.add_scalar('test_mpjpe', mpjpe, epoch)

                # save normalized pose, not org pose
                if not args.smoke_test:
                    save_path = os.path.join(final_output_dir, 'last_samples.npz')
                    logger.info(f'save eval samples to {save_path}')
                    np.savez(save_path,
                        **{
                            'pred3d': trajs[:, :20, None, :, :],  # [t, b, 1, j, 3]
                            'gt3d': labels_3d[:20, :, :].cpu().numpy()[None, :, None, ...]  # [1, b, 1, j, 3]
                        }
                    )

                # log and save ckpt
                logger.info(f'Save checkpoint to {final_output_dir}')
                save_dict = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': current_optimizer.state_dict(),
                    'ema': state['ema'].state_dict(),
                    'step': state['step'],
                }
                torch.save(save_dict, os.path.join(final_output_dir, 'checkpoint.pth'))

                if mpjpe < best_error:
                    # best checkpoint under my metric
                    best_error = mpjpe
                    logger.info(f'★ NEW BEST: {best_error:.2f}mm ★')
                    torch.save(
                        {
                            'model_state_dict': model.state_dict(),
                            'epoch': epoch + 1,
                            'ema': state['ema'].state_dict(),
                            'step': state['step'],
                        },
                        os.path.join(final_output_dir, 'best_model.pth')
                    )
            # lr_scheduler.step()
    except Exception as e:
        traceback.print_exc()
    finally:
        writer.close()
        if args.smoke_test:
            logger.info("="*50)
            logger.info("SMOKE TEST COMPLETED")
            logger.info(f"Final MPJPE: {mpjpe:.2f}mm")
            logger.info("="*50)
        logger.info(f'End. Final output dir: {final_output_dir}')


if __name__ == '__main__':
    app.run(main, flags_parser=parse_args)