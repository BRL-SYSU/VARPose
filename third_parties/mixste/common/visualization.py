# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import subprocess as sp
import os
import cv2
import pandas as pd

def get_resolution(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        for line in pipe.stdout:
            w, h = line.decode().strip().split(',')
            return int(w), int(h)


def get_fps(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', filename]
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        for line in pipe.stdout:
            a, b = line.decode().strip().split('/')
            return int(a) / int(b)


def read_video(filename, skip=0, limit=-1):
    w, h = get_resolution(filename)
    # w = 1000
    # h = 1002

    command = ['ffmpeg',
               '-i', filename,
               '-f', 'image2pipe',
               '-pix_fmt', 'rgb24',
               '-vsync', '0',
               '-vcodec', 'rawvideo', '-']

    i = 0
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        while True:
            data = pipe.stdout.read(w * h * 3)
            if not data:
                break
            i += 1
            if i > limit and limit != -1:
                continue
            if i > skip:
                yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))


def downsample_tensor(X, factor):
    length = X.shape[0] // factor * factor
    return np.mean(X[:length].reshape(-1, factor, *X.shape[1:]), axis=1)


def render_animation(keypoints, keypoints_metadata, poses, skeleton, fps, bitrate, azim, output, viewport,
                     limit=-1, downsample=1, size=6, input_video_path=None, input_video_skip=0, newpose=None,
                     save_frames=False, frame_dir=None, frame_idx=None, single_frame_image_path:str=None):
    """
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
     -- 'save_frames': if True, save each frame as an individual image file.
     -- 'frame_dir': directory to save individual frame images.
    """
    plt.ioff()
    if newpose is not None:
        fig = plt.figure(figsize=(size * (1 + len(poses) + len(newpose)), size))
        ax_in = fig.add_subplot(1, 1 + len(poses) + len(newpose), 1)
    else:
        fig = plt.figure(figsize=(size * (1 + len(poses)), size))
        ax_in = fig.add_subplot(1, 1 + len(poses), 1)
    
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()
    ax_in.set_title('Input')

    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7
    if newpose is not None:
        axnew = fig.add_subplot(1, 1 + len(poses) + len(newpose), 2, projection='3d')
        axnew.view_init(elev=15., azim=azim)
        axnew.set_xlim3d([-radius / 2, radius / 2])
        axnew.set_zlim3d([0, radius])
        axnew.set_ylim3d([-radius / 2, radius / 2])
        try:
            axnew.set_aspect('equal')
        except NotImplementedError:
            axnew.set_aspect('auto')
        axnew.set_xticklabels([])
        axnew.set_yticklabels([])
        axnew.set_zticklabels([])
        axnew.dist = 7.5
        axnew.set_title('PoseFormer') #, pad=35
        ax_3d.append(axnew)
        lines_3d.append([])
        trajectories.append(newpose[:, 0, [0, 1]])

    for index, (title, data) in enumerate(poses.items()):
        ax = fig.add_subplot(1, 1 + len(poses), index + 2, projection='3d')
        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius / 2, radius / 2])
        try:
            ax.set_aspect('equal')
        except NotImplementedError:
            ax.set_aspect('auto')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 7.5
        ax.set_title(title) #, pad=35
        ax_3d.append(ax)
        lines_3d.append([])
        trajectories.append(data[:, 0, [0, 1]])
    poses = list(poses.values())

    # Decode video
    if input_video_path is None:
        # Black background
        all_frames = np.zeros((keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
    else:
        # Load video using ffmpeg
        all_frames = []
        for f in read_video(input_video_path, skip=input_video_skip, limit=limit):
            all_frames.append(f)
        effective_length = min(keypoints.shape[0], len(all_frames))
        all_frames = all_frames[:effective_length]

        keypoints = keypoints[input_video_skip:] # todo remove
        for idx in range(len(poses)):
            poses[idx] = poses[idx][input_video_skip:]
        if newpose is not None:
            newpose = newpose[input_video_skip:]

        if fps is None:
            fps = get_fps(input_video_path)
    
    if single_frame_image_path is not None:
        single_frame_image = cv2.imread(single_frame_image_path)
        single_frame_image = cv2.cvtColor(single_frame_image, cv2.COLOR_BGR2RGB)

    if downsample > 1:
        keypoints = downsample_tensor(keypoints, downsample)
        all_frames = downsample_tensor(np.array(all_frames), downsample).astype('uint8')
        if newpose is not None:
            newpose = downsample_tensor(newpose, downsample)
            for idx in range(len(poses)+len(newpose)):
                poses[idx] = downsample_tensor(poses[idx], downsample)
                trajectories[idx] = downsample_tensor(trajectories[idx], downsample)
        else:
            for idx in range(len(poses)):
                poses[idx] = downsample_tensor(poses[idx], downsample)
                trajectories[idx] = downsample_tensor(trajectories[idx], downsample)
        
        fps /= downsample

    initialized = False
    image = None
    lines = []
    points = None

    if limit < 1:
        limit = len(all_frames)
    else:
        limit = min(limit, len(all_frames))

    parents = skeleton.parents()
    def update_video(i):
        nonlocal initialized, image, lines, points

        for n, ax in enumerate(ax_3d):
            ax.set_xlim3d([-radius/2 + trajectories[n][i, 0], radius/2 + trajectories[n][i, 0]])
            ax.set_ylim3d([-radius/2 + trajectories[n][i, 1], radius/2 + trajectories[n][i, 1]])

        # Update 2D poses
        joints_right_2d = keypoints_metadata['keypoints_symmetry'][1]
        colors_2d = np.full(keypoints.shape[1], 'blue')
        if not initialized:
            if single_frame_image_path is not None and input_video_path is None:
                image = ax_in.imshow(single_frame_image, aspect='equal')
            else:
                image = ax_in.imshow(all_frames[i], aspect='equal')

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
                    # Draw skeleton only if keypoints match (otherwise we don't have the parents definition)
                    lines.append(ax_in.plot([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                            [keypoints[i, j, 1], keypoints[i, j_parent, 1]], color='pink'))

                col = 'red' if j in skeleton.joints_right() else 'black'
                
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                               [pos[j, 1], pos[j_parent, 1]],
                                               [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))
            # Plot 2D keypoints
            points = ax_in.scatter(*keypoints[i].T, 10, color=colors_2d, edgecolors='white', zorder=10)

            initialized = True
        else:
            if single_frame_image_path is not None and input_video_path is None:
                image.set_data(single_frame_image)
            else:
                image.set_data(all_frames[i])

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
                    lines[j-1][0].set_data([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                             [keypoints[i, j, 1], keypoints[i, j_parent, 1]])

                # Plot 2D keypoints
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n][j-1][0].set_xdata(np.array([pos[j, 0], pos[j_parent, 0]]))
                    lines_3d[n][j-1][0].set_ydata(np.array([pos[j, 1], pos[j_parent, 1]]))
                    lines_3d[n][j-1][0].set_3d_properties(np.array([pos[j, 2], pos[j_parent, 2]]), zdir='z')
            # Plot 2D keypoints
            points.set_offsets(keypoints[i])

        # Save frame if requested
        if save_frames and frame_dir is not None and frame_idx>=0 and i==frame_idx:
            # Create directory if it doesn't exist
            os.makedirs(frame_dir, exist_ok=True)
            # Save frame with zero-padded index
            video_name = os.path.splitext(os.path.basename(output))[0]
            frame_path = os.path.join(frame_dir, f'frame_{i:06d}_{video_name}.png')
            plt.savefig(frame_path, dpi=600, bbox_inches='tight')

        print('{}/{}      '.format(i, limit), end='\r')


    fig.tight_layout()

    anim = FuncAnimation(fig, update_video, frames=np.arange(0, limit), interval=1000/fps, repeat=False)
    if output.endswith('.mp4'):
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    elif output.endswith('.gif'):
        anim.save(output, dpi=80, writer='imagemagick')
    else:
        raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
    plt.close()


def plot_scale_and_effect(csv_paths, metrics=None, output_dir='log/scale_search/plots'):
    """
    Plot metric curves against scale for different weights, with all metrics
    combined in one figure.
    
    Args:
        csv_paths: List[str] - result CSV paths for different weights.
        metrics: List[str] - metric names to plot; defaults to ['MPJPE', 'P-MPJPE'].
        output_dir: str - output directory; defaults to 'log/scale_search/plots'.
    """
    if metrics is None:
        metrics = ['MPJPE', 'P-MPJPE']
    
    # Create output directory.
    os.makedirs(output_dir, exist_ok=True)
    
    # Read all CSV data.
    data_list = []
    labels = []
    
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        data_list.append(df)
        
        # Extract a label from the filename (weight configuration identifier).
        # Example: scale_search_results.csv -> "results"
        # scale_search_results_no_dense.csv -> "results_no_dense"
        filename = os.path.basename(csv_path)
        # Remove the .csv extension and the scale_search_ prefix.
        label = filename.replace('.csv', '').replace('scale_search_', '')
        if not label:
            label = 'default'
        labels.append(label)
    
    # Create subplots: one row with multiple columns.
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))
    
    # If there is only one metric, axes is not an array and needs special handling.
    if num_metrics == 1:
        axes = [axes]
    
    # Define colors and line styles.
    colors = plt.cm.tab10(np.linspace(0, 1, len(csv_paths)))
    line_styles = ['-', '--', '-.', ':']
    
    # Plot one subplot for each metric.
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Collect the scale values at the minimum point of each curve.
        min_scales = []
        
        # Plot data from each CSV file.
        for csv_idx, (df, label) in enumerate(zip(data_list, labels)):
            # Find the minimum value for this metric.
            min_value = df[metric].min()
            min_idx = df[metric].idxmin()
            min_scale = df.loc[min_idx, 'Scale']
            min_scales.append(min_scale)
            
            # Add the minimum value to the legend.
            label_with_min = f"{label} ({min_value:.1f})"
            
            ax.plot(df['Scale'], df[metric], 
                   marker='o', 
                   linestyle=line_styles[csv_idx % len(line_styles)],
                   color=colors[csv_idx],
                   label=label_with_min,
                   linewidth=2,
                   markersize=6)
        
        # Adjust the x-axis range to show only the local region around minima.
        if min_scales:
            min_scale = min(min_scales)
            x_min = min_scale - 0.5
            x_max = min_scale + 0.5
            ax.set_xlim(x_min, x_max)
            
            # Collect y-values for all curves within the x-axis range.
            y_values_in_range = []
            for df in data_list:
                # Select data points within the x-axis range.
                mask = (df['Scale'] >= x_min) & (df['Scale'] <= x_max)
                y_values_in_range.extend(df[mask][metric].values)
            
            # Set the y-axis range based on y-values in range.
            if y_values_in_range:
                y_min = min(y_values_in_range)
                y_max = max(y_values_in_range)
                # Add a 10% margin for better readability.
                y_margin = (y_max - y_min) * 0.1
                ax.set_ylim(y_min - y_margin, y_max + y_margin)
        
        # Show a legend for each subplot.
        ax.legend(loc='best', fontsize=10)
        
        # Set subplot style.
        ax.set_xlabel('Scale', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric} vs Scale', fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Adjust layout.
    plt.tight_layout()
    
    # Save figure.
    output_path = os.path.join(output_dir, 'scale_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to: {output_path}")
