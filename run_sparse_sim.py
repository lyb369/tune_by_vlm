import os
import argparse
from typing import Optional

import numpy as np

from sparse_sim import (
    load_image_gray,
    save_image_gray,
    estimate_gaussian_kernel_params,
    gaussian_kernel,
    save_kernel_heatmap,
    save_blur_effect_grid,
    compute_metrics,
    infer_pair_paths,
)
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'sparse-deconv-py-main'))
from sparse_recon.sparse_deconv import sparse_deconv
from qwen_helper import QwenLocal, ensure_dirs, load_memory, append_memory, build_history_summaries


def main():
    parser = argparse.ArgumentParser(description='Sparse-SIM restoration with Qwen-guided parameter search')
    parser.add_argument('--gt_path', type=str, default='images')
    parser.add_argument('--blur_path', type=str, default='images')
    # parser.add_argument('--pick_blur', type=str, default=None, help='Specific blurred filename to use')
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--fidelity_init', type=float, default=150.0)
    parser.add_argument('--sparsity_init', type=float, default=10.0)
    parser.add_argument('--use_demo_params', action='store_true', help='Use demo.py parameters for faster execution')
    parser.add_argument('--model_path', type=str, default=os.environ.get('QWEN_MODEL_PATH', 'Qwen2.5-7B-Instruct'))
    parser.add_argument('--outputs_dir', type=str, default='outputs_deconv')
    parser.add_argument('--memory_dir', type=str, default='memory_deconv')
    parser.add_argument('--test_mode', action='store_true', help='测试模式：不使用Qwen模型，仅使用启发式参数调整')
    args = parser.parse_args()

    ensure_dirs([args.outputs_dir, args.memory_dir])
    
    gt_path = args.gt_path
    blur_path = args.blur_path

    # gt_path, blur_path = infer_pair_paths(args.images_dir, args.pick_blur)
    gt = load_image_gray(gt_path)
    f = load_image_gray(blur_path)

    if not args.use_demo_params:
        # Only estimate PSF when not using demo parameters
        best_sigma, best_size, best_b = estimate_gaussian_kernel_params(gt, f)
        k = gaussian_kernel(best_sigma, best_size)
        
        # Debug: print sigma values for comparison
        demo_sigma = 280 / 65  # demo.py sigma
        print(f"Debug - best_sigma: {best_sigma:.4f}, demo.py sigma: {demo_sigma:.4f}, ratio: {best_sigma/demo_sigma:.2f}x")

        # Visualize kernel and its effect
        save_kernel_heatmap(os.path.join(args.outputs_dir, f'kernel_sigma{best_sigma:.2f}_k{best_size}.png'), k)
        save_blur_effect_grid(os.path.join(args.outputs_dir, 'kernel_effect_grid.png'), gt, k, best_b)
    else:
        # Use demo parameters - no PSF estimation needed
        best_sigma = 280 / 65  # demo.py sigma
        best_size = 11  # reasonable default
        best_b = 0.0   # reasonable default
        print(f"Using demo.py parameters - sigma: {best_sigma:.4f}")

    records = load_memory(args.memory_dir)
    qwen = QwenLocal(args.model_path)
    
    # Load model once at the beginning
    print("Loading Qwen model...")
    qwen.load()
    print("Qwen model loaded successfully!")

    # Initial params
    fidelity = args.fidelity_init
    sparsity = args.sparsity_init

    x = f.copy()
    for it in range(1, args.rounds + 1):
        # Optimize with current params using sparse-deconv
        if args.use_demo_params:
            # Use demo.py parameters for faster execution
            sigma_val = 280 / 65  # demo.py sigma
            x = sparse_deconv(
                img=f,
                sigma=sigma_val,
                sparse_iter=100,
                fidelity=fidelity,  # demo.py fidelity
                sparsity=sparsity,   # demo.py sparsity
                tcontinuity=0.5,
                background=2,  # demo.py background
                deconv_iter=7,
                deconv_type=1,  # demo.py deconv_type
                up_sample=0,
            )
        else:
            # Use estimated parameters (slower but more adaptive)
            x = sparse_deconv(
                img=f,
                sigma=[best_sigma, best_sigma],
                sparse_iter=100,
                fidelity=fidelity,
                sparsity=sparsity,
                tcontinuity=0.5,
                background=2,
                deconv_iter=7,
                deconv_type=1,
                up_sample=0,
            )

        # Ensure CPU NumPy array for downstream save/metrics
        try:
            import cupy as cp  # type: ignore
            if hasattr(x, "__cuda_array_interface__"):
                x_cpu = cp.asnumpy(x)
            else:
                x_cpu = x
        except Exception:
            x_cpu = x

        # Save image
        out_path = os.path.join(args.outputs_dir, f'iter_{it:03d}.png')
        save_image_gray(out_path, x_cpu)

        # Metrics (FRC & rolling FRC)
        metrics = compute_metrics(x_cpu, gt)

        # Append memory
        rec = {
            'iteration': it,
            'fidelity': f"{fidelity:.9f}",
            'sparsity': f"{sparsity:.9f}",
            'kernel': {'sigma': float(best_sigma), 'size': int(best_size), 'bias_b': float(best_b)},
            'use_demo_params': args.use_demo_params,
            'metrics': metrics,
            'gt_path': gt_path,
            'blur_path': blur_path,
            'restored_path': out_path,
        }
        append_memory(args.memory_dir, it, rec)
        records.append(rec)

        # Build observation text for Qwen
        obs = (
            f"已完成第{it}轮: FRC_AUC={metrics.get('frc_auc', 0.0):.4f}, R-FRC_AUC_mean={metrics.get('rfrc_auc_mean', 0.0):.4f}, "
            f"halfbit_res={metrics.get('frc_halfbit_res', 0.0):.4f}, mean_grad={metrics.get('mean_grad_mag', 0.0):.6f}。"
            f" 当前参数: fidelity={fidelity:.6f}, sparsity={sparsity:.6f}。"
            f" 请给出更优的参数(fidelity, sparsity)，目标是提升FRC_AUC/R-FRC_AUC与halfbit_res。"
        )
        histories = build_history_summaries(records)
        
        # 保存旧参数用于比较
        old_fidelity, old_sparsity = fidelity, sparsity
        
        if args.test_mode:
            print(f"\n=== 第{it}轮参数优化（测试模式） ===")
            print(f"当前参数: fidelity={fidelity:.6f}, sparsity={sparsity:.6f}")
            print(f"当前指标: FRC_AUC={metrics.get('frc_auc', 0.0):.4f}, R-FRC_AUC_mean={metrics.get('rfrc_auc_mean', 0.0):.4f}, halfbit_res={metrics.get('frc_halfbit_res', 0.0):.4f}")
            print("使用启发式参数调整...")
            
            # 启发式参数调整策略
            if it == 1:
                fidelity = old_fidelity * 0.8  # 降低fidelity，增加去噪
                sparsity = old_sparsity * 1.2  # 增加sparsity
            elif it == 2:
                fidelity = old_fidelity * 1.2  # 增加fidelity，保持细节
                sparsity = old_sparsity * 0.8  # 降低sparsity
            else:
                # 基于FRC_AUC变化调整
                if it > 1 and records[-2]['metrics'].get('frc_auc', 0.0) < metrics.get('frc_auc', 0.0):
                    # 指标提升，继续当前方向
                    fidelity = old_fidelity * 1.1
                    sparsity = old_sparsity * 0.9
                else:
                    # 指标下降，反向调整
                    fidelity = old_fidelity * 0.9
                    sparsity = old_sparsity * 1.1
            
            print(f"启发式调整后: fidelity={fidelity:.6f}, sparsity={sparsity:.6f}")
            
        else:
            try:
                print(f"\n=== 第{it}轮参数优化 ===")
                print(f"当前参数: fidelity={fidelity:.6f}, sparsity={sparsity:.6f}")
                print(f"当前指标: FRC_AUC={metrics.get('frc_auc', 0.0):.4f}, R-FRC_AUC_mean={metrics.get('rfrc_auc_mean', 0.0):.4f}, halfbit_res={metrics.get('frc_halfbit_res', 0.0):.4f}")
                print("正在调用Qwen模型获取参数建议...")
                
                fidelity, sparsity = qwen.suggest_params(histories, obs)
                
                print(f"Qwen建议参数: fidelity={fidelity:.6f}, sparsity={sparsity:.6f}")
                
                # 检查参数是否真的改变了
                if abs(fidelity - old_fidelity) < 1e-6 and abs(sparsity - old_sparsity) < 1e-6:
                    print("警告: Qwen模型返回的参数与当前参数相同，可能模型没有正确响应")
                    # 使用简单的启发式规则来调整参数
                    if it == 1:
                        fidelity = old_fidelity * 0.8  # 降低fidelity
                        sparsity = old_sparsity * 1.2  # 增加sparsity
                    elif it == 2:
                        fidelity = old_fidelity * 1.2  # 增加fidelity
                        sparsity = old_sparsity * 0.8  # 降低sparsity
                    else:
                        # 基于FRC_AUC变化调整
                        if it > 1 and records[-2]['metrics'].get('frc_auc', 0.0) < metrics.get('frc_auc', 0.0):
                            fidelity = old_fidelity * 1.1  # 指标提升，继续增加fidelity
                            sparsity = old_sparsity * 0.9
                        else:
                            fidelity = old_fidelity * 0.9  # 指标下降，降低fidelity
                            sparsity = old_sparsity * 1.1
                    
                    print(f"使用启发式调整: fidelity={fidelity:.6f}, sparsity={sparsity:.6f}")
                
            except Exception as e:
                print(f"Qwen模型调用失败: {e}")
                print("使用启发式参数调整...")
                
                # 简单的启发式参数调整
                if it == 1:
                    fidelity = old_fidelity * 0.8
                    sparsity = old_sparsity * 1.2
                elif it == 2:
                    fidelity = old_fidelity * 1.2
                    sparsity = old_sparsity * 0.8
                else:
                    # 基于历史性能调整（FRC_AUC）
                    if len(records) >= 2:
                        prev_auc = records[-2]['metrics'].get('frc_auc', 0.0)
                        if metrics.get('frc_auc', 0.0) > prev_auc:
                            fidelity = old_fidelity * 1.1
                            sparsity = old_sparsity * 0.9
                        else:
                            fidelity = old_fidelity * 0.9
                            sparsity = old_sparsity * 1.1
                    else:
                        fidelity = old_fidelity * 0.9
                        sparsity = old_sparsity * 1.1
                
                print(f"启发式调整后: fidelity={fidelity:.6f}, sparsity={sparsity:.6f}")
        
        print(f"下一轮将使用: fidelity={fidelity:.6f}, sparsity={sparsity:.6f}")
        print("=" * 50)

    print('完成参数搜索与图像恢复。结果已保存到输出与记忆目录。')


if __name__ == '__main__':
    main()


