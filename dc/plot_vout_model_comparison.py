#!/usr/bin/env python3
"""
Compare VOUT node voltage evolution:
- Left: Without DNN (direct DC analysis)
- Right: With DNN initial guess from trained models (W400, W800, or single model)
Generate animated GIF showing iteration-by-iteration convergence

Usage:
    python dc/plot_vout_model_comparison.py --config plot_config.yaml
"""

import json
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

def load_config(config_path):
    """加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        dict: 配置字典
    """
    if config_path is None:
        config_path = Path(__file__).parent / "plot_config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"⚠️  Warning: Config file not found: {config_path}")
        print(f"   Using default configuration (file paths must be provided)")
        return None
    
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ Config loaded from: {config_path.name}\n")
        return config
    except ImportError:
        print(f"⚠️  Warning: PyYAML not installed, skipping config file")
        print(f"   Install with: pip install pyyaml")
        return None
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None


def validate_json_files(json_files_dict):
    """验证JSON文件存在性
    
    Args:
        json_files_dict: {'no_dnn': path, 'w400': path, 'w800': path}
    
    Returns:
        dict: 验证后的有效文件路径和缺失文件列表
    """
    valid_files = {}
    missing_files = []
    
    for key, path in json_files_dict.items():
        if path is None or path == '':
            missing_files.append(key)
            valid_files[key] = None
        else:
            if os.path.exists(path):
                valid_files[key] = path
                print(f"✅ Found {key} JSON: {path}")
            else:
                print(f"❌ Missing {key} JSON: {path}")
                missing_files.append(key)
                valid_files[key] = None
    
    return valid_files, missing_files


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
    - Right: With DNN (可能包含W400和/或W800模型)
    
    Args:
        json_no_dnn: 无DNN的JSON路径（必需）
        json_w400: W400模型的JSON路径（可选，为None时跳过）
        json_w800: W800模型的JSON路径（可选，为None时跳过）
        output_png: 输出PNG文件路径
    """
    # Parse no_dnn (always required)
    vout_no_dnn, iterations_no_dnn, conv_no_dnn, final_no_dnn = parse_vout_from_json(json_no_dnn)
    
    # Parse W400 and W800 (optional)
    vout_w400 = iterations_w400 = conv_w400 = final_w400 = None
    vout_w800 = iterations_w800 = conv_w800 = final_w800 = None
    
    if json_w400 is not None:
        vout_w400, iterations_w400, conv_w400, final_w400 = parse_vout_from_json(json_w400)
    
    if json_w800 is not None:
        vout_w800, iterations_w800, conv_w800, final_w800 = parse_vout_from_json(json_w800)
    
    print(f"\n=== Model Comparison ===")
    print(f"No DNN: {len(vout_no_dnn)} data points, converged={conv_no_dnn}, final VOUT={final_no_dnn:.6f}V")
    if json_w400 is not None:
        print(f"W400 model: {len(vout_w400)} data points, converged={conv_w400}, final VOUT={final_w400:.6f}V")
    else:
        print(f"W400 model: NOT PROVIDED (skipped)")
    if json_w800 is not None:
        print(f"W800 model: {len(vout_w800)} data points, converged={conv_w800}, final VOUT={final_w800:.6f}V")
    else:
        print(f"W800 model: NOT PROVIDED (skipped)")
    
    print(f"Initial VOUT - No DNN: {vout_no_dnn[0]:.6f}V")
    if json_w400 is not None:
        print(f"Initial VOUT - W400 model: {vout_w400[0]:.6f}V")
    if json_w800 is not None:
        print(f"Initial VOUT - W800 model: {vout_w800[0]:.6f}V")
    
    # Create figure with two subplots
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Calculate target value (average of available final values)
    available_finals = [final_no_dnn]
    if final_w400 is not None:
        available_finals.append(final_w400)
    if final_w800 is not None:
        available_finals.append(final_w800)
    target_value = sum(available_finals) / len(available_finals)
    
    # Common y-axis range
    all_vout = vout_no_dnn.copy()
    if vout_w400 is not None:
        all_vout.extend(vout_w400)
    if vout_w800 is not None:
        all_vout.extend(vout_w800)
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
    
    # ===== RIGHT SUBPLOT: With DNN (available models) =====
    has_w400 = vout_w400 is not None
    has_w800 = vout_w800 is not None
    
    if has_w400:
        iterations_display_w400 = list(range(len(vout_w400)))
        ax_right.plot(iterations_display_w400, vout_w400, 
                      'o-', linewidth=3, markersize=7,
                      color='#FF6B6B', label=f'W400 Model',
                      alpha=0.85)
    
    if has_w800:
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
    
    # Dynamic title based on available models
    if has_w400 and has_w800:
        right_title = f'With DNN Initial Guess (W400 vs W800)\nIterations: {len(vout_w400)}'
        title_color = '#4ECDC4'
    elif has_w800:
        right_title = f'With DNN Initial Guess (W800 Model)\nIterations: {len(vout_w800)}'
        title_color = '#4ECDC4'
    elif has_w400:
        right_title = f'With DNN Initial Guess (W400 Model)\nIterations: {len(vout_w400)}'
        title_color = '#FF6B6B'
    else:
        right_title = 'No DNN models provided'
        title_color = '#999999'
    
    ax_right.set_title(right_title, fontsize=13, fontweight='bold', color=title_color)
    ax_right.legend(fontsize=11, loc='best', framealpha=0.95)
    ax_right.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax_right.tick_params(labelsize=11)
    ax_right.set_ylim(y_min, y_max)
    
    plt.suptitle('VOUT Convergence Comparison: Effect of DNN Initial Guess on NMOS', 
                 fontsize=17, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to: {output_png}")
    plt.close()



def generate_frames_and_gif(json_no_dnn, json_w400, json_w800, frames_dir, gif_path, fps=4):
    """
    Generate animation frames and GIF showing progressive convergence
    Left side: No DNN iterations
    Right side: W400 and/or W800 model iterations (independent)
    
    Args:
        json_no_dnn: 无DNN的JSON路径（必需）
        json_w400: W400模型的JSON路径（可选）
        json_w800: W800模型的JSON路径（可选）
        frames_dir: 输出帧目录
        gif_path: 输出GIF路径
        fps: 帧率
    """
    # Parse no_dnn (always required)
    vout_no_dnn, iterations_no_dnn, conv_no_dnn, final_no_dnn = parse_vout_from_json(json_no_dnn)
    
    # Parse W400 and W800 (optional)
    vout_w400 = vout_w800 = None
    if json_w400 is not None:
        vout_w400, iterations_w400, conv_w400, final_w400 = parse_vout_from_json(json_w400)
    
    if json_w800 is not None:
        vout_w800, iterations_w800, conv_w800, final_w800 = parse_vout_from_json(json_w800)
    
    # Create frames directory
    os.makedirs(frames_dir, exist_ok=True)
    
    # Total frames = max iterations from no_dnn (left side drives animation length)
    total_frames = len(vout_no_dnn)
    
    print(f"\nGenerating {total_frames} animation frames...")
    print(f"Left side (No DNN): {len(vout_no_dnn)} iterations")
    if vout_w400 is not None:
        print(f"Right side (DNN W400): {len(vout_w400)} iterations")
    else:
        print(f"Right side (DNN W400): NOT PROVIDED")
    if vout_w800 is not None:
        print(f"Right side (DNN W800): {len(vout_w800)} iterations")
    else:
        print(f"Right side (DNN W800): NOT PROVIDED")
    
    # Calculate target value
    available_finals = [final_no_dnn]
    if vout_w400 is not None:
        available_finals.append(final_w400)
    if vout_w800 is not None:
        available_finals.append(final_w800)
    target_value = sum(available_finals) / len(available_finals)
    
    # Common y-axis range
    all_vout = vout_no_dnn.copy()
    if vout_w400 is not None:
        all_vout.extend(vout_w400)
    if vout_w800 is not None:
        all_vout.extend(vout_w800)
    y_min = min(all_vout) - 0.15
    y_max = max(all_vout) + 0.15
    
    frame_files = []
    
    for frame_idx in range(total_frames):
        # Left side: progressive iteration
        iter_no_dnn_current = frame_idx + 1
        
        # Right side: independent iteration (finishes early, then plateaus)
        iter_w400_current = None
        iter_w800_current = None
        
        if vout_w400 is not None:
            iter_w400_current = min(frame_idx + 1, len(vout_w400))
        if vout_w800 is not None:
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
        has_w400 = vout_w400 is not None
        has_w800 = vout_w800 is not None
        
        if has_w400:
            x_w400 = list(range(iter_w400_current))
            y_w400 = vout_w400[:iter_w400_current]
            
            if len(x_w400) > 0:
                ax_right.plot(x_w400, y_w400, 
                              'o-', linewidth=3, markersize=7,
                              color='#FF6B6B', label='W400 Model',
                              alpha=0.85)
        
        if has_w800:
            x_w800 = list(range(iter_w800_current))
            y_w800 = vout_w800[:iter_w800_current]
            
            if len(x_w800) > 0:
                ax_right.plot(x_w800, y_w800, 
                              's-', linewidth=3, markersize=7,
                              color='#4ECDC4', label='W800 Model',
                              alpha=0.85)
        
        ax_right.axhline(y=target_value, color='#2ECC71', linestyle='--', 
                         linewidth=2.5, label=f'Target: {target_value:.4f}V', alpha=0.7)
        
        ax_right.set_xlabel('Iteration Number', fontsize=12, fontweight='bold')
        ax_right.set_ylabel('VOUT Voltage (V)', fontsize=12, fontweight='bold')
        
        # Show iteration status
        if has_w400 and has_w800:
            iter_status = f'W400: {iter_w400_current}/{len(vout_w400)}, W800: {iter_w800_current}/{len(vout_w800)}'
            title_color = '#4ECDC4'
        elif has_w800:
            iter_status = f'W800: {iter_w800_current}/{len(vout_w800)}'
            title_color = '#4ECDC4'
        elif has_w400:
            iter_status = f'W400: {iter_w400_current}/{len(vout_w400)}'
            title_color = '#FF6B6B'
        else:
            iter_status = 'No DNN models'
            title_color = '#999999'
        
        ax_right.set_title(f'With DNN Initial Guess\n{iter_status}',
                           fontsize=13, fontweight='bold', color=title_color)
        ax_right.legend(fontsize=11, loc='best', framealpha=0.95)
        ax_right.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax_right.tick_params(labelsize=11)
        ax_right.set_ylim(y_min, y_max)
        ax_right.set_xlim(-0.5, len(vout_no_dnn) - 0.5)
        
        plt.suptitle(f'VOUT Convergence Comparison: Effect of DNN Initial Guess on NMOS', 
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
    parser = argparse.ArgumentParser(
        description="Compare VOUT voltage evolution across different DNN models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default config file (plot_config.yaml in same directory)
  python plot_vout_model_comparison.py
  
  # Use custom config file
  python plot_vout_model_comparison.py --config /path/to/config.yaml
  
  # Command line overrides (optional)
  python plot_vout_model_comparison.py --json-no-dnn /path/to/file.json --json-w800 /path/to/file.json
        """
    )
    
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration YAML file (default: plot_config.yaml)")
    parser.add_argument("--json-no-dnn", type=str, default=None,
                       help="Path to no-DNN JSON file (overrides config)")
    parser.add_argument("--json-w400", type=str, default=None,
                       help="Path to W400 model JSON file (overrides config, optional)")
    parser.add_argument("--json-w800", type=str, default=None,
                       help="Path to W800 model JSON file (overrides config, optional)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for all outputs (overrides config)")
    parser.add_argument("--skip-gif", action="store_true",
                       help="Skip GIF generation, only generate static plot")
    parser.add_argument("--skip-static", action="store_true",
                       help="Skip static plot generation, only generate GIF")
    
    args = parser.parse_args()
    
    # Load configuration from YAML
    config = load_config(args.config)
    
    # Prepare file paths (command line args take precedence over config)
    json_files = {}
    output_config = {}
    viz_config = {}
    
    if config:
        # From config file
        data_cfg = config.get('data', {})
        json_files = {
            'no_dnn': data_cfg.get('json_no_dnn'),
            'w400': data_cfg.get('json_w400'),
            'w800': data_cfg.get('json_w800'),
        }
        
        output_cfg = config.get('output', {})
        output_config = {
            'output_png': output_cfg.get('output_png'),
            'frames_dir': output_cfg.get('frames_dir'),
            'gif_path': output_cfg.get('gif_path'),
        }
        
        viz_cfg = config.get('visualization', {})
        viz_config = {
            'fps': viz_cfg.get('fps', 8),
            'dpi': viz_cfg.get('dpi', 300),
            'generate_gif': viz_cfg.get('generate_gif', True),
            'generate_static': viz_cfg.get('generate_static', True),
        }
    
    # Override with command line arguments
    if args.json_no_dnn:
        json_files['no_dnn'] = args.json_no_dnn
    if args.json_w400:
        json_files['w400'] = args.json_w400
    if args.json_w800:
        json_files['w800'] = args.json_w800
    
    # Validate required files
    if not json_files.get('no_dnn'):
        print("❌ ERROR: json_no_dnn is required but not provided")
        print("   Provide it via config file or --json-no-dnn argument")
        exit(1)
    
    if not (json_files.get('w400') or json_files.get('w800')):
        print("⚠️  WARNING: At least one of json_w400 or json_w800 should be provided for comparison")
        print("   Continuing with no-DNN analysis only...")
    
    # Check file existence
    print("📂 Checking input files...")
    json_files, missing = validate_json_files(json_files)
    
    if missing:
        if 'no_dnn' in missing:
            print(f"❌ ERROR: Required file json_no_dnn is missing")
            exit(1)
        else:
            print(f"⚠️  Some optional files are missing: {missing}")
    
    print()
    
    # Set output paths
    if args.output_dir:
        output_config['output_png'] = os.path.join(args.output_dir, 'vout_model_comparison.png')
        output_config['frames_dir'] = os.path.join(args.output_dir, 'gif_model_cmp')
        output_config['gif_path'] = os.path.join(args.output_dir, 'vout_model_comparison.gif')
    
    # Use defaults if not specified
    if not output_config.get('output_png'):
        output_config['output_png'] = './vout_model_comparison.png'
    if not output_config.get('frames_dir'):
        output_config['frames_dir'] = './gif_model_cmp'
    if not output_config.get('gif_path'):
        output_config['gif_path'] = './vout_model_comparison.gif'
    
    # Handle skip flags
    generate_gif = viz_config.get('generate_gif', True)
    generate_static = viz_config.get('generate_static', True)
    
    if args.skip_gif:
        generate_gif = False
    if args.skip_static:
        generate_static = False
    
    if not generate_gif and not generate_static:
        print("❌ ERROR: Both GIF and static plot generation are disabled")
        exit(1)
    
    fps = viz_config.get('fps', 8)
    
    # Create output directory if needed
    output_dir_path = Path(output_config['output_png']).parent
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Output Configuration:")
    print(f"  Static plot: {output_config['output_png']}")
    print(f"  GIF frames:  {output_config['frames_dir']}")
    print(f"  GIF output:  {output_config['gif_path']}")
    print(f"  Frame rate:  {fps} fps")
    print()
    
    # Generate visualizations
    try:
        if generate_static:
            print("="*60)
            print("Generating VOUT model comparison plot...")
            print("="*60)
            plot_vout_model_comparison(
                json_files['no_dnn'],
                json_files['w400'],
                json_files['w800'],
                output_config['output_png']
            )
        
        if generate_gif:
            print("\n" + "="*60)
            print("Generating animated GIF...")
            print("="*60)
            generate_frames_and_gif(
                json_files['no_dnn'],
                json_files['w400'],
                json_files['w800'],
                output_config['frames_dir'],
                output_config['gif_path'],
                fps=fps
            )
        
        print("\n" + "="*60)
        print("✅ All done!")
        print("="*60)
        if generate_static:
            print(f"Static plot: {output_config['output_png']}")
        if generate_gif:
            print(f"GIF animation: {output_config['gif_path']}")
            print(f"Frames directory: {output_config['frames_dir']}")
    
    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        exit(1)
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

