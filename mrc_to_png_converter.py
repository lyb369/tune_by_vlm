#!/usr/bin/env python3
"""
MRC文件转PNG转换器
将MRC文件（可能包含多个切片）转换为PNG文件，通过像素平均处理多个切片
"""

import os
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_mrc_file(file_path):
    """
    读取MRC文件
    MRC文件格式：前1024字节是头部信息，包含维度信息
    """
    try:
        with open(file_path, 'rb') as f:
            # 读取头部信息（前1024字节）
            header = np.frombuffer(f.read(1024), dtype=np.int32)
            
            # 从头部获取维度信息
            nx, ny, nz = header[0], header[1], header[2]
            
            # 获取数据类型
            mode = header[3]
            if mode == 0:
                dtype = np.int8
            elif mode == 1:
                dtype = np.int16
            elif mode == 2:
                dtype = np.float32
            elif mode == 6:
                dtype = np.uint16
            else:
                dtype = np.float32  # 默认使用float32
            
            # 读取数据
            f.seek(1024)  # 跳过头部
            data = np.frombuffer(f.read(), dtype=dtype)
            
            # 重塑数据为3D数组
            if nz > 1:
                data = data.reshape((nz, ny, nx))
            else:
                data = data.reshape((ny, nx))
            
            return data, (nx, ny, nz)
            
    except Exception as e:
        logger.error(f"读取MRC文件失败 {file_path}: {e}")
        return None, None

def normalize_image(data):
    """
    将图像数据标准化到0-255范围
    """
    # 处理负值和异常值
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 如果数据全为0，返回全黑图像
    if np.max(data) == np.min(data):
        return np.zeros_like(data, dtype=np.uint8)
    
    # 标准化到0-1范围
    data_min = np.min(data)
    data_max = np.max(data)
    normalized = (data - data_min) / (data_max - data_min)
    
    # 转换为0-255范围
    return (normalized * 255).astype(np.uint8)

def convert_mrc_to_png(mrc_path, output_path):
    """
    将单个MRC文件转换为PNG文件
    """
    logger.info(f"正在处理: {mrc_path}")
    
    # 读取MRC文件
    data, dimensions = read_mrc_file(mrc_path)
    if data is None:
        return False
    
    nx, ny, nz = dimensions
    
    # 如果是3D数据（多个切片），计算像素平均
    if len(data.shape) == 3:
        logger.info(f"检测到3D数据，包含 {nz} 个切片，正在计算像素平均...")
        # 计算所有切片的像素平均
        averaged_data = np.mean(data, axis=0)
    else:
        # 2D数据直接使用
        averaged_data = data
    
    # 标准化图像数据
    normalized_data = normalize_image(averaged_data)
    
    # 转换为PIL图像并保存
    try:
        image = Image.fromarray(normalized_data, mode='L')  # 'L'表示灰度图像
        image.save(output_path)
        logger.info(f"成功保存: {output_path}")
        return True
    except Exception as e:
        logger.error(f"保存PNG文件失败 {output_path}: {e}")
        return False

def process_directory(input_dir, output_dir=None, recursive=True):
    """
    处理目录中的所有MRC文件
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        return
    
    # 如果没有指定输出目录，在输入目录下创建png_output文件夹
    if output_dir is None:
        output_path = input_path / "png_output"
    else:
        output_path = Path(output_dir)
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有MRC文件
    if recursive:
        mrc_files = list(input_path.rglob("*.mrc"))
    else:
        mrc_files = list(input_path.glob("*.mrc"))
    
    logger.info(f"找到 {len(mrc_files)} 个MRC文件")
    
    success_count = 0
    failed_count = 0
    
    for mrc_file in mrc_files:
        # 计算相对路径以保持目录结构
        relative_path = mrc_file.relative_to(input_path)
        png_file = output_path / relative_path.with_suffix('.png')
        
        # 创建输出子目录
        png_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换文件
        if convert_mrc_to_png(mrc_file, png_file):
            success_count += 1
        else:
            failed_count += 1
    
    logger.info(f"转换完成！成功: {success_count}, 失败: {failed_count}")

def main():
    parser = argparse.ArgumentParser(description='将MRC文件转换为PNG文件')
    parser.add_argument('input_dir', help='输入目录路径')
    parser.add_argument('-o', '--output', help='输出目录路径（可选）')
    parser.add_argument('--no-recursive', action='store_true', help='不递归处理子目录')
    
    args = parser.parse_args()
    
    # 处理目录
    process_directory(
        args.input_dir, 
        args.output, 
        recursive=not args.no_recursive
    )

if __name__ == "__main__":
    # 如果直接运行脚本，处理当前目录
    if len(os.sys.argv) == 1:
        logger.info("未指定参数，处理当前目录...")
        process_directory(".", recursive=True)
    else:
        main()
