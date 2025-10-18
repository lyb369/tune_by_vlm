#!/usr/bin/env python3
"""
图像区域选择工具
用于可视化选择RawSIMData_level图像中的256x256区域，
并自动提取对应的512x512 SIM_gt区域
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
import logging
from typing import Tuple, Optional, List
import json

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageRegionSelector:
    def __init__(self, data_dir: str, output_dir: str = None):
        """
        初始化图像区域选择器
        
        Args:
            data_dir: 包含所有细胞数据的根目录
            output_dir: 输出目录，如果为None则在data_dir下创建selected_regions
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / "selected_regions"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 当前状态
        self.current_cell = None
        self.current_level_file = None
        self.current_sim_gt_file = None
        self.current_image = None
        self.current_sim_gt = None
        self.selection_start = None
        self.selection_end = None
        self.selection_active = False
        
        # 获取所有需要处理的文件
        self.cell_files = self._get_cell_files()
        self.current_index = 0
        
        # 保存选择记录
        self.selections = []
        
    def _get_cell_files(self) -> List[dict]:
        """获取所有需要处理的细胞文件"""
        cell_files = []
        
        # 遍历所有子目录
        for subdir in self.data_dir.iterdir():
            if subdir.is_dir():
                # 遍历每个细胞目录
                for cell_dir in subdir.iterdir():
                    if cell_dir.is_dir() and cell_dir.name.startswith('Cell_'):
                        # 检查不同的目录结构
                        cell_info = self._check_cell_structure(cell_dir, subdir.name)
                        if cell_info:
                            cell_files.append(cell_info)
        
        logger.info(f"找到 {len(cell_files)} 个细胞需要处理")
        return cell_files
    
    def _check_cell_structure(self, cell_dir: Path, subdir_name: str) -> Optional[dict]:
        """检查细胞目录结构并返回相应的文件信息"""
        
        # 结构1: 直接包含RawSIMData_level_01.png和SIM_gt文件 (CCPs, F-actin等)
        level_file = cell_dir / "RawSIMData_level_01.png"
        if level_file.exists():
            sim_gt_files = list(cell_dir.glob("SIM_gt*.png"))
            if sim_gt_files:
                return {
                    'cell_path': cell_dir,
                    'level_file': level_file,
                    'sim_gt_files': sim_gt_files,
                    'subdir': subdir_name,
                    'cell_name': cell_dir.name,
                    'structure_type': 'direct'
                }
        
        # 结构2: ER目录结构 - 包含RawSIMData子目录
        raw_sim_data_dir = cell_dir / "RawSIMData"
        if raw_sim_data_dir.exists() and raw_sim_data_dir.is_dir():
            level_file = raw_sim_data_dir / "RawSIMData_level_01.png"
            if level_file.exists():
                # 查找GTSIM目录作为对应的SIM_gt
                gtsim_dir = cell_dir / "GTSIM"
                if gtsim_dir.exists() and gtsim_dir.is_dir():
                    sim_gt_files = list(gtsim_dir.glob("GTSIM_level_*.png"))
                    if sim_gt_files:
                        return {
                            'cell_path': cell_dir,
                            'level_file': level_file,
                            'sim_gt_files': sim_gt_files,
                            'subdir': subdir_name,
                            'cell_name': cell_dir.name,
                            'structure_type': 'er'
                        }
        
        return None
    
    def _load_current_images(self):
        """加载当前细胞的所有图像"""
        if self.current_index >= len(self.cell_files):
            return False
            
        cell_info = self.cell_files[self.current_index]
        self.current_cell = cell_info
        
        # 加载level图像
        self.current_image = cv2.imread(str(cell_info['level_file']))
        if self.current_image is None:
            logger.error(f"无法加载图像: {cell_info['level_file']}")
            return False
        
        # 选择合适的SIM_gt图像
        if cell_info['structure_type'] == 'er':
            # ER目录结构：选择GTSIM_level_01.png
            self.current_sim_gt_file = None
            for sim_file in cell_info['sim_gt_files']:
                if 'level_01' in sim_file.name:
                    self.current_sim_gt_file = sim_file
                    break
            if self.current_sim_gt_file is None:
                self.current_sim_gt_file = cell_info['sim_gt_files'][0]
        else:
            # 直接结构：选择第一个SIM_gt文件
            self.current_sim_gt_file = cell_info['sim_gt_files'][0]
        
        self.current_sim_gt = cv2.imread(str(self.current_sim_gt_file))
        if self.current_sim_gt is None:
            logger.error(f"无法加载SIM_gt图像: {self.current_sim_gt_file}")
            return False
        
        logger.info(f"加载图像: {cell_info['cell_name']} - {cell_info['subdir']}")
        return True
    
    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selection_start = (x, y)
            self.selection_active = True
            self.selection_end = None
            
        elif event == cv2.EVENT_MOUSEMOVE and self.selection_active:
            self.selection_end = (x, y)
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.selection_end = (x, y)
            self.selection_active = False
    
    def _draw_selection(self, image):
        """在图像上绘制选择框"""
        display_image = image.copy()
        
        if self.selection_start and self.selection_end:
            # 确保选择框是256x256
            x1, y1 = self.selection_start
            x2, y2 = self.selection_end
            
            # 计算256x256的边界
            size = 256
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # 确保不超出图像边界
            half_size = size // 2
            x1 = max(0, center_x - half_size)
            y1 = max(0, center_y - half_size)
            x2 = min(image.shape[1], center_x + half_size)
            y2 = min(image.shape[0], center_y + half_size)
            
            # 绘制选择框
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 显示尺寸信息
            cv2.putText(display_image, f"Selection: {x2-x1}x{y2-y1}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return display_image
    
    def _save_selection(self):
        """保存当前选择"""
        if not self.selection_start or not self.selection_end:
            logger.warning("没有选择区域")
            return False
        
        # 计算256x256区域
        x1, y1 = self.selection_start
        x2, y2 = self.selection_end
        
        size = 256
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        half_size = size // 2
        x1 = max(0, center_x - half_size)
        y1 = max(0, center_y - half_size)
        x2 = min(self.current_image.shape[1], center_x + half_size)
        y2 = min(self.current_image.shape[0], center_y + half_size)
        
        # 提取level图像的256x256区域
        level_region = self.current_image[y1:y2, x1:x2]
        
        # 计算对应的SIM_gt区域（2倍分辨率）
        sim_x1 = x1 * 2
        sim_y1 = y1 * 2
        sim_x2 = x2 * 2
        sim_y2 = y2 * 2
        
        # 确保不超出SIM_gt图像边界
        sim_x2 = min(self.current_sim_gt.shape[1], sim_x2)
        sim_y2 = min(self.current_sim_gt.shape[0], sim_y2)
        
        sim_region = self.current_sim_gt[sim_y1:sim_y2, sim_x1:sim_x2]
        
        # 生成文件名
        cell_info = self.current_cell
        timestamp = len(self.selections) + 1
        level_filename = f"{cell_info['subdir']}_{cell_info['cell_name']}_level_{timestamp:03d}.png"
        sim_filename = f"{cell_info['subdir']}_{cell_info['cell_name']}_sim_{timestamp:03d}.png"
        
        # 保存文件
        level_path = self.output_dir / level_filename
        sim_path = self.output_dir / sim_filename
        
        cv2.imwrite(str(level_path), level_region)
        cv2.imwrite(str(sim_path), sim_region)
        
        # 记录选择信息
        selection_info = {
            'cell': cell_info['cell_name'],
            'subdir': cell_info['subdir'],
            'level_file': str(cell_info['level_file']),
            'sim_gt_file': str(self.current_sim_gt_file),
            'level_region': [int(x1), int(y1), int(x2), int(y2)],
            'sim_region': [int(sim_x1), int(sim_y1), int(sim_x2), int(sim_y2)],
            'level_output': str(level_path),
            'sim_output': str(sim_path),
            'timestamp': timestamp
        }
        
        self.selections.append(selection_info)
        
        logger.info(f"保存选择: {level_filename} 和 {sim_filename}")
        return True
    
    def _next_image(self):
        """切换到下一张图像"""
        self.current_index += 1
        if self.current_index >= len(self.cell_files):
            logger.info("所有图像处理完成！")
            self._save_selection_log()
            return False
        return True
    
    def _save_selection_log(self):
        """保存选择记录到JSON文件"""
        log_path = self.output_dir / "selection_log.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.selections, f, indent=2, ensure_ascii=False)
        logger.info(f"选择记录已保存到: {log_path}")
    
    def run(self):
        """运行可视化选择工具"""
        if not self.cell_files:
            logger.error("没有找到需要处理的文件")
            return
        
        # 创建窗口
        window_name = "Image Region Selector"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        # 加载第一张图像
        if not self._load_current_images():
            logger.error("无法加载图像")
            return
        
        logger.info("操作说明:")
        logger.info("- 鼠标左键拖拽选择256x256区域")
        logger.info("- 按 's' 保存当前选择")
        logger.info("- 按 'n' 切换到下一张图像")
        logger.info("- 按 'q' 退出程序")
        
        while True:
            # 显示当前图像
            display_image = self._draw_selection(self.current_image)
            
            # 添加信息文本
            cell_info = self.current_cell
            info_text = f"{cell_info['subdir']} - {cell_info['cell_name']} ({self.current_index + 1}/{len(self.cell_files)})"
            cv2.putText(display_image, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.putText(display_image, "Press 's' to save, 'n' for next, 'q' to quit", 
                       (10, display_image.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow(window_name, display_image)
            
            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self._save_selection()
            elif key == ord('n'):
                if not self._next_image():
                    break
                if not self._load_current_images():
                    break
                # 重置选择状态
                self.selection_start = None
                self.selection_end = None
                self.selection_active = False
        
        cv2.destroyAllWindows()
        self._save_selection_log()
        logger.info(f"处理完成！共保存了 {len(self.selections)} 个选择")

def main():
    parser = argparse.ArgumentParser(description='图像区域选择工具')
    parser.add_argument('data_dir', help='包含所有细胞数据的根目录')
    parser.add_argument('-o', '--output', help='输出目录（可选）')
    
    args = parser.parse_args()
    
    selector = ImageRegionSelector(args.data_dir, args.output)
    selector.run()

if __name__ == "__main__":
    # 如果直接运行，使用默认路径
    if len(os.sys.argv) == 1:
        data_dir = r"C:\Users\admin\Desktop\dataset\output"
        logger.info(f"使用默认数据目录: {data_dir}")
        selector = ImageRegionSelector(data_dir)
        selector.run()
    else:
        main()
