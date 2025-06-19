#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
行走检测阈值调整工具
帮助用户找到最佳的行走检测参数
"""

import cv2
import numpy as np
import math
from ultralytics import YOLO
import argparse

def matrix_distance(point_a, point_b):
    """计算两点间距离"""
    x1, y1 = point_a
    x2, y2 = point_b
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def test_threshold(video_path, base_threshold, sample_frames=100):
    """测试特定阈值的效果"""
    print(f"测试阈值: {base_threshold}")
    
    # 加载模型
    model = YOLO('yolov8n-pose.pt')
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return 0, 0
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 计算实际阈值
    scale_factor = min(width, height) / 720
    actual_threshold = int(base_threshold * scale_factor)
    
    # 初始化脚部位置记录
    L_feet = []
    R_feet = []
    
    # 统计信息
    frames_with_feet = 0
    frames_with_walk = 0
    
    # 采样处理（为了加快速度）
    frame_interval = max(1, total_frames // sample_frames)
    
    frame_count = 0
    processed_count = 0
    
    while cap.isOpened() and processed_count < sample_frames:
        success, frame = cap.read()
        if not success:
            break
            
        frame_count += 1
        
        # 只处理采样帧
        if frame_count % frame_interval != 0:
            continue
            
        processed_count += 1
        
        results = model(frame, save=False, conf=0.6)
        
        if len(results[0].boxes) < 1:
            continue
        
        # 处理检测到的人
        boxes = results[0].boxes[0]
        keypoints = results[0].keypoints[0]
        key = np.array(keypoints.xy.cpu().numpy())
        
        if np.sum(key) > 0:
            # 检查脚部关键点
            KEY = [False] * 17
            for i in range(len(key[0])):
                if key[0][i][1] + key[0][i][0] == 0:
                    KEY[i] = False
                else:
                    KEY[i] = True
            
            # 更新脚部位置
            if KEY[15] and KEY[16]:
                frames_with_feet += 1
                L_feetxy = [int(key[0][15][0]), int(key[0][15][1])]
                R_feetxy = [int(key[0][16][0]), int(key[0][16][1])]
                
                # 记录脚部位置历史
                if len(L_feet) < 3:
                    L_feet.append(L_feetxy)
                    R_feet.append(R_feetxy)
                else:
                    L_feet.pop(0)
                    R_feet.pop(0)
                    L_feet.append(L_feetxy)
                    R_feet.append(R_feetxy)
                
                # 检查行走
                if len(L_feet) == 3 and len(R_feet) == 3:
                    # 计算移动距离
                    left_move1 = matrix_distance(L_feet[0], L_feet[1])
                    left_move2 = matrix_distance(L_feet[1], L_feet[2])
                    right_move1 = matrix_distance(R_feet[0], R_feet[1])
                    right_move2 = matrix_distance(R_feet[1], R_feet[2])
                    
                    # 判断行走
                    if ((left_move1 > actual_threshold and left_move2 > actual_threshold) or 
                        (right_move1 > actual_threshold and right_move2 > actual_threshold)):
                        frames_with_walk += 1
    
    cap.release()
    
    walk_rate = frames_with_walk / frames_with_feet * 100 if frames_with_feet > 0 else 0
    print(f"  实际阈值: {actual_threshold}, 行走检测率: {walk_rate:.1f}% ({frames_with_walk}/{frames_with_feet})")
    
    return walk_rate, frames_with_walk

def find_optimal_threshold(video_path):
    """找到最佳阈值"""
    print(f"开始为视频 {video_path} 寻找最佳阈值...")
    
    # 测试不同的基础阈值
    thresholds = [3, 5, 8, 10, 12, 15, 18, 20, 25]
    results = {}
    
    for threshold in thresholds:
        walk_rate, walk_count = test_threshold(video_path, threshold)
        results[threshold] = {
            'walk_rate': walk_rate,
            'walk_count': walk_count
        }
    
    # 分析结果
    print("\n" + "="*60)
    print("阈值测试结果汇总")
    print("="*60)
    
    best_threshold = None
    best_score = 0
    
    for threshold, result in results.items():
        walk_rate = result['walk_rate']
        walk_count = result['walk_count']
        
        # 评分标准：检测率在10%-50%之间，且检测次数适中
        if 10 <= walk_rate <= 50 and walk_count > 0:
            score = walk_count * (1 - abs(walk_rate - 30) / 30)  # 30%为理想检测率
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        print(f"阈值 {threshold:2d}: 检测率 {walk_rate:5.1f}%, 检测次数 {walk_count:3d}")
    
    print("\n" + "="*60)
    if best_threshold:
        print(f"推荐阈值: {best_threshold}")
        print(f"预期检测率: {results[best_threshold]['walk_rate']:.1f}%")
        print(f"预期检测次数: {results[best_threshold]['walk_count']}")
    else:
        print("未找到合适的阈值，建议手动调整")
        print("如果检测率过低，尝试降低阈值")
        print("如果检测率过高，尝试提高阈值")
    
    return best_threshold, results

def generate_code_snippet(best_threshold):
    """生成代码片段"""
    print("\n" + "="*60)
    print("代码修改建议")
    print("="*60)
    
    print("在 yolov8pose.py 中，将以下行：")
    print("base_threshold = 10")
    print("修改为：")
    print(f"base_threshold = {best_threshold}")
    
    print("\n或者，您可以在代码中添加动态调整：")
    print("""
# 根据视频类型动态调整阈值
if 'testvideo' in video_path.lower():
    base_threshold = """ + str(best_threshold) + """
else:
    base_threshold = 10
""")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='行走检测阈值调整工具')
    parser.add_argument('video_path', help='视频文件路径')
    parser.add_argument('--sample-frames', type=int, default=100, help='采样帧数（默认100）')
    
    args = parser.parse_args()
    
    # 寻找最佳阈值
    best_threshold, results = find_optimal_threshold(args.video_path)
    
    # 生成代码修改建议
    if best_threshold:
        generate_code_snippet(best_threshold)
    
    print("\n使用说明:")
    print("1. 运行此工具分析您的视频")
    print("2. 根据推荐阈值修改代码")
    print("3. 重新测试视频效果")
    print("4. 如果效果仍不理想，可以进一步微调阈值") 