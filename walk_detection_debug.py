#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
行走检测调试工具
用于分析行走检测的问题和优化参数
"""

import cv2
import numpy as np
import math
from ultralytics import YOLO
import time

def matrix_distance(point_a, point_b):
    """计算两点间距离"""
    x1, y1 = point_a
    x2, y2 = point_b
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def analyze_walk_detection(video_path, output_path=None):
    """分析视频中的行走检测"""
    print(f"开始分析视频: {video_path}")
    
    # 加载模型
    model = YOLO('yolov8n-pose.pt')
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频信息:")
    print(f"  - 分辨率: {width}x{height}")
    print(f"  - FPS: {fps}")
    print(f"  - 总帧数: {total_frames}")
    
    # 初始化脚部位置记录
    L_feet = []
    R_feet = []
    L_feetxy = [0, 0]
    R_feetxy = [0, 0]
    
    # 动态阈值计算
    base_threshold = 10
    scale_factor = min(width, height) / 720
    lengTH = int(base_threshold * scale_factor)
    
    print(f"检测参数:")
    print(f"  - 基础阈值: {base_threshold}")
    print(f"  - 缩放因子: {scale_factor:.2f}")
    print(f"  - 最终阈值: {lengTH}")
    
    # 统计信息
    total_frames_processed = 0
    frames_with_person = 0
    frames_with_feet = 0
    frames_with_walk = 0
    walk_detections = []
    
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame_count += 1
        total_frames_processed += 1
        
        # 每10帧打印一次进度
        if frame_count % 10 == 0:
            print(f"处理进度: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
        
        results = model(frame, save=False, conf=0.6)
        
        if len(results[0].boxes) < 1:
            continue
            
        frames_with_person += 1
        
        # 处理检测到的人
        boxes = results[0].boxes[0]
        keypoints = results[0].keypoints[0]
        key = np.array(keypoints.xy.cpu().numpy())
        
        if np.sum(key) > 0:
            # 检查脚部关键点 (15: 左脚踝, 16: 右脚踝)
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
                    if ((left_move1 > lengTH and left_move2 > lengTH) or 
                        (right_move1 > lengTH and right_move2 > lengTH)):
                        frames_with_walk += 1
                        walk_info = {
                            'frame': frame_count,
                            'left_move1': left_move1,
                            'left_move2': left_move2,
                            'right_move1': right_move1,
                            'right_move2': right_move2,
                            'threshold': lengTH
                        }
                        walk_detections.append(walk_info)
                        print(f"帧 {frame_count}: 检测到行走 - 左脚移动: {left_move1:.1f}, {left_move2:.1f}, 右脚移动: {right_move1:.1f}, {right_move2:.1f}")
            else:
                L_feetxy = [0, 0]
                R_feetxy = [0, 0]
    
    cap.release()
    
    # 输出分析结果
    print("\n" + "="*50)
    print("行走检测分析结果")
    print("="*50)
    print(f"总处理帧数: {total_frames_processed}")
    print(f"检测到人的帧数: {frames_with_person}")
    print(f"检测到脚部的帧数: {frames_with_feet}")
    print(f"检测到行走的帧数: {frames_with_walk}")
    print(f"行走检测率: {frames_with_walk/frames_with_person*100:.2f}% (基于有人的帧)")
    print(f"行走检测率: {frames_with_walk/frames_with_feet*100:.2f}% (基于有脚部的帧)")
    
    if walk_detections:
        print(f"\n行走检测详情 (共{len(walk_detections)}次):")
        for i, detection in enumerate(walk_detections[:10]):  # 只显示前10次
            print(f"  {i+1}. 帧 {detection['frame']}: 左脚({detection['left_move1']:.1f}, {detection['left_move2']:.1f}) 右脚({detection['right_move1']:.1f}, {detection['right_move2']:.1f})")
        if len(walk_detections) > 10:
            print(f"  ... 还有 {len(walk_detections) - 10} 次检测")
    
    # 建议优化参数
    print(f"\n优化建议:")
    if frames_with_walk == 0:
        print("  - 未检测到行走，建议降低阈值")
        print(f"  - 尝试将阈值从 {lengTH} 降低到 {max(1, lengTH//2)}")
    elif frames_with_walk < frames_with_person * 0.1:
        print("  - 行走检测率较低，建议降低阈值")
        print(f"  - 当前阈值: {lengTH}, 建议尝试: {max(1, lengTH//2)}")
    else:
        print("  - 行走检测正常")
    
    return walk_detections

def test_different_thresholds(video_path):
    """测试不同阈值的效果"""
    print("测试不同阈值的效果...")
    
    thresholds = [5, 8, 10, 12, 15, 20]
    results = {}
    
    for threshold in thresholds:
        print(f"\n测试阈值: {threshold}")
        # 这里可以修改阈值并重新运行检测
        # 为了简化，这里只是示例
        results[threshold] = f"阈值 {threshold} 的检测结果"
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python walk_detection_debug.py <视频文件路径>")
        print("示例: python walk_detection_debug.py testvideo.flv")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # 分析行走检测
    walk_detections = analyze_walk_detection(video_path)
    
    # 测试不同阈值
    # test_different_thresholds(video_path) 