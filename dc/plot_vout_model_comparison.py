#!/usr/bin/env python3
"""
Compare VOUT node voltage evolution:
- Left: Without DNN (direct DC analysis)
- Right: With DNN initial guess from W400 and W800 trained models
Generate animated GIF showing iteration-by-iteration convergence
"""

import json
import os
import matplotlib.pyplot as plt
from PIL import Image

def parse_vout_from_json(json_file):
    """
    Parse Newton iteration JSON and extract VOUT node voltage progression
    
    Returns: (vout_voltages, iteration_numbers, converged, final_value)
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract metadata
    metadata = data.get('metadata', {})
    converged = metadata.get('converged', False)
    
    # Extract VOUT voltages and iteration numbers from all iterations
    iterations = data.get('iterations', [])
    vout_voltages = []
    iteration_numbers = []
    
    for iter_data in iterations:
        node_names = iter_data.get('node_names', [])
        x = iter_data.get('x', [])
        iter_num = iter_data.get('iteration', None)
        
        # Find VOUT index
        if 'VOUT' in node_names and iter_num is not None:
            vout_idx = node_names.index('VOUT')
            if vout_idx < len(x):
                vout_voltages.append(x[vout_idx])
                # Map: -1 -> 0, 0 -> 1, 1 -> 2, etc.
                iteration_numbers.append(iter_num + 1)
    
    final_value = vout_voltages[-1] if vout_voltages else None
    
    return vout_voltages, iteration_numbers, converged, final_value


def plot_vout_model_comparison(json_no_dnn, json_w400, json_w800, output_png):
    """
    Plot VOUT voltage evolution in two subplots:
    - Left: Without DNN
    - Right: With DNN (W400 vs W800 models)
    """
    # Parse all three JSON files
    vout_no_dnn, iterations_no_dnn, conv_no_dnn, final_no_dnn = parse_vout_from_json(json_no_dnn)
    vout_w400, iterations_w400, conv_w400, final_w400 = parse_vout_from_json(json_w400)
    vout_w800, iterations_w800, conv_w800, final_w800 = parse_vout_from_json(json_w800)
    
    print(f"\n=== Model Comparison ===")
    print(f"No DNN: {len(vout_no_dnn)} data points, converged={conv_no_dnn}, final VOUT={final_no_dnn:.6f}V")
    print(f"W400 model: {len(vout_w400)} data points, converged={conv_w400}, final VOUT={final_w400:.6f}V")
    print(f"W800 model: {len(vout_w800)} data points, converged={conv_w800}, final VOUT={final_w800:.6f}V")
    print(f"Initial VOUT - No DNN: {vout_no_dnn[0]:.6f}V")
    print(f"Initial VOUT - W400 model: {vout_w400[0]:.6f}V")
    print(f"Initial VOUT - W800 model: {vout_w800[0]:.6f}V")
    
    # Create figure with two subplots
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Target value (average of all final values)
    target_value = (final_no_dnn + final_w400 + final_w800) / 3
    
    # Common y-axis range
    all_vout = vout_no_dnn + vout_w400 + vout_w800
    y_min = min(all_vout) - 0.15
    y_max = max(all_vout) + 0.15
    
    # ===== LEFT SUBPLOT: Without DNN =====
    iterations_display_no_dnn = list(range(len(vout_no_dnn)))
    ax_left.plot(iterations_display_no_dnn, vout_no_dnn, 
                 'o-', linewidth=3, markersize=6,
                 color='black', label='VOUT Voltage',
                 alpha=0.85)
    
    # Target line
    ax_left.axhline(y=target_value, color='#2ECC71', linestyle='--', 
                    linewidth=2.5, label=f'Target: {target_value:.4f}V', alpha=0.7)
    
    ax_left.set_xlabel('Iteration Number', fontsize=12, fontweight='bold')
    ax_left.set_ylabel('VOUT Voltage (V)', fontsize=12, fontweight='bold')
    ax_left.set_title(f'Without DNN Initial Guess\nIterations: {len(vout_no_dnn)}', 
                      fontsize=13, fontweight='bold', color='black')
    ax_left.legend(fontsize=11, loc='best', framealpha=0.95)
    ax_left.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax_left.tick_params(labelsize=11)
    ax_left.set_ylim(y_min, y_max)
    
    # ===== RIGHT SUBPLOT: With DNN (W400 vs W800) =====
    iterations_display_w400 = list(range(len(vout_w400)))
    ax_right.plot(iterations_display_w400, vout_w400, 
                  'o-', linewidth=3, markersize=7,
                  color='#FF6B6B', label=f'W400 Model',
                  alpha=0.85)
    
    iterations_display_w800 = list(range(len(vout_w800)))
    ax_right.plot(iterations_display_w800, vout_w800, 
                  's-', linewidth=3, markersize=7,
                  color='#4ECDC4', label=f'W800 Model',
                  alpha=0.85)
    
    # Target line
    ax_right.axhline(y=target_value, color='#2ECC71', linestyle='--', 
                     linewidth=2.5, label=f'Target: {target_value:.4f}V', alpha=0.7)
    
    ax_right.set_xlabel('Iteration Number', fontsize=12, fontweight='bold')
    ax_right.set_ylabel('VOUT Voltage (V)', fontsize=12, fontweight='bold')
    ax_right.set_title(f'With DNN Initial Guess (W400 vs W800)\nIterations: {len(vout_w400)}', 
                       fontsize=13, fontweight='bold', color='#4ECDC4')
    ax_right.legend(fontsize=11, loc='best', framealpha=0.95)
    ax_right.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax_right.tick_params(labelsize=11)
    ax_right.set_ylim(y_min, y_max)
    
    plt.suptitle('VOUT Convergence Comparison: Effect of DNN Initial Guess on NMOS W800', 
                 fontsize=17, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to: {output_png}")
    plt.close()


def generate_frames_and_gif(json_no_dnn, json_w400, json_w800, frames_dir, gif_path, fps=4):
    """
    Generate animation frames and GIF showing progressive convergence
    Left side: No DNN iterations (independent)
    Right side: W400 and W800 model iterations (independent, finishes early)
    """
    # Parse all three JSON files
    vout_no_dnn, iterations_no_dnn, conv_no_dnn, final_no_dnn = parse_vout_from_json(json_no_dnn)
    vout_w400, iterations_w400, conv_w400, final_w400 = parse_vout_from_json(json_w400)
    vout_w800, iterations_w800, conv_w800, final_w800 = parse_vout_from_json(json_w800)
    
    # Create frames directory
    os.makedirs(frames_dir, exist_ok=True)
    
    # Total frames = max iterations from no_dnn (left side drives animation length)
    total_frames = len(vout_no_dnn)
    max_iter_dnn = max(len(vout_w400), len(vout_w800))
    
    print(f"\nGenerating {total_frames} animation frames...")
    print(f"Left side (No DNN): {len(vout_no_dnn)} iterations")
    print(f"Right side (DNN W400): {len(vout_w400)} iterations")
    print(f"Right side (DNN W800): {len(vout_w800)} iterations")
    
    target_value = (final_no_dnn + final_w400 + final_w800) / 3
    
    # Common y-axis range
    all_vout = vout_no_dnn + vout_w400 + vout_w800
    y_min = min(all_vout) - 0.15
    y_max = max(all_vout) + 0.15
    
    frame_files = []
    
    for frame_idx in range(total_frames):
        # Left side: progressive iteration
        iter_no_dnn_current = frame_idx + 1
        
        # Right side: independent iteration (finishes early, then plateaus)
        iter_w400_current = min(frame_idx + 1, len(vout_w400))
        iter_w800_current = min(frame_idx + 1, len(vout_w800))
        
        # Create figure
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 7))
        
        # ===== LEFT SUBPLOT: No DNN (progressive) =====
        x_no_dnn = list(range(iter_no_dnn_current))
        y_no_dnn = vout_no_dnn[:iter_no_dnn_current]
        
        ax_left.plot(x_no_dnn, y_no_dnn, 
                     'o-', linewidth=3, markersize=6,
                     color='black', label='VOUT Voltage',
                     alpha=0.85)
        
        ax_left.axhline(y=target_value, color='#2ECC71', linestyle='--', 
                        linewidth=2.5, label=f'Target: {target_value:.4f}V', alpha=0.7)
        
        ax_left.set_xlabel('Iteration Number', fontsize=12, fontweight='bold')
        ax_left.set_ylabel('VOUT Voltage (V)', fontsize=12, fontweight='bold')
        ax_left.set_title(f'Without DNN Initial Guess\nIteration: {iter_no_dnn_current}/{len(vout_no_dnn)}',
                          fontsize=13, fontweight='bold', color='black')
        ax_left.legend(fontsize=11, loc='best', framealpha=0.95)
        ax_left.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax_left.tick_params(labelsize=11)
        ax_left.set_ylim(y_min, y_max)
        ax_left.set_xlim(-0.5, len(vout_no_dnn) - 0.5)
        
        # ===== RIGHT SUBPLOT: With DNN (progressive, independent) =====
        x_w400 = list(range(iter_w400_current))
        y_w400 = vout_w400[:iter_w400_current]
        
        x_w800 = list(range(iter_w800_current))
        y_w800 = vout_w800[:iter_w800_current]
        
        if len(x_w400) > 0:
            ax_right.plot(x_w400, y_w400, 
                          'o-', linewidth=3, markersize=7,
                          color='#FF6B6B', label='W400 Model',
                          alpha=0.85)
        
        if len(x_w800) > 0:
            ax_right.plot(x_w800, y_w800, 
                          's-', linewidth=3, markersize=7,
                          color='#4ECDC4', label='W800 Model',
                          alpha=0.85)
        
        ax_right.axhline(y=target_value, color='#2ECC71', linestyle='--', 
                         linewidth=2.5, label=f'Target: {target_value:.4f}V', alpha=0.7)
        
        ax_right.set_xlabel('Iteration Number', fontsize=12, fontweight='bold')
        ax_right.set_ylabel('VOUT Voltage (V)', fontsize=12, fontweight='bold')
        
        # Show iteration status for both models
        iter_status = f'W400: {iter_w400_current}/{len(vout_w400)}, W800: {iter_w800_current}/{len(vout_w800)}'
        ax_right.set_title(f'With DNN Initial Guess\n{iter_status}',
                           fontsize=13, fontweight='bold', color='#4ECDC4')
        ax_right.legend(fontsize=11, loc='best', framealpha=0.95)
        ax_right.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax_right.tick_params(labelsize=11)
        ax_right.set_ylim(y_min, y_max)
        ax_right.set_xlim(-0.5, len(vout_no_dnn) - 0.5)
        
        plt.suptitle(f'VOUT Convergence Comparison: Effect of DNN Initial Guess on NMOS W800', 
                     fontsize=17, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save frame
        frame_file = os.path.join(frames_dir, f'frame_{frame_idx+1:03d}.png')
        plt.savefig(frame_file, dpi=300, bbox_inches='tight')
        frame_files.append(frame_file)
        plt.close()
        
        if (frame_idx + 1) % 10 == 0 or frame_idx == total_frames - 1:
            print(f"  Generated frame {frame_idx + 1}/{total_frames}")
    
    # Create GIF from frames
    print(f"\nCreating GIF animation...")
    images = [Image.open(f) for f in frame_files]
    duration = int(1000 / fps)  # milliseconds per frame
    
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=True
    )
    
    print(f"✓ GIF animation saved to: {gif_path}")
    print(f"  Total frames: {total_frames}")
    print(f"  Frame rate: {fps} fps")
    print(f"  Duration: {duration} ms per frame")


if __name__ == '__main__':
    # File paths
    json_no_dnn = '/home/hu/saratoga/dc/logs/20260204_150739/nmos/newton_analysis/newton_dc_nmos_20260204_150745.json'
    json_w400 = '/home/hu/saratoga/dc/logs/20260204_235148/nmos/newton_analysis/newton_dc_nmos_20260204_235154.json'
    json_w800 = '/home/hu/saratoga/dc/logs/20260204_235447/nmos/newton_analysis/newton_dc_nmos_20260204_235453.json'
    output_png = '/home/hu/saratoga/dc/vout_model_comparison.png'
    frames_dir = '/home/hu/saratoga/dc/gif_model_cmp'
    gif_path = '/home/hu/saratoga/dc/vout_model_comparison.gif'
    
    # Check if files exist
    if not os.path.exists(json_no_dnn):
        print(f"ERROR: No DNN file not found: {json_no_dnn}")
        exit(1)
    
    if not os.path.exists(json_w400):
        print(f"ERROR: W400 model file not found: {json_w400}")
        exit(1)
    
    if not os.path.exists(json_w800):
        print(f"ERROR: W800 model file not found: {json_w800}")
        exit(1)
    
    # Generate static comparison plot
    print("Generating VOUT model comparison plot...")
    plot_vout_model_comparison(json_no_dnn, json_w400, json_w800, output_png)
    
    # Generate animated GIF
    print("\n" + "="*60)
    generate_frames_and_gif(json_no_dnn, json_w400, json_w800, frames_dir, gif_path, fps=8)
    
    print("\n" + "="*60)
    print("All done!")
    print(f"Static plot: {output_png}")
    print(f"GIF animation: {gif_path}")
    print(f"Frames directory: {frames_dir}")
