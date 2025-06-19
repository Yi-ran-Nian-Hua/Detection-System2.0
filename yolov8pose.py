import os
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,QHBoxLayout, QPushButton, QLabel, QFrame, QFileDialog)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QPalette,QImage,QPixmap

from ultralytics import YOLO
import threading
import math
import cv2
import numpy as np
import time

class CameraInterface(QMainWindow):
    def __init__(self):
        global frameyplopose #视频图片
        global L_Hand #举左手计数
        global R_Hand #举右手计数
        global sit_Time #坐姿计时
        global stand_Time #站姿计时
        global walk_Time #行走计时
        # 新增线程安全的共享变量
        self.stand_status = False  # 站立状态
        self.sit_status = False    # 坐姿状态
        self.walk_status = False   # 行走状态
        self.left_hand_up = False  # 左手举起状态
        self.right_hand_up = False # 右手举起状态
        self.TimeBool = False #计时状态
        
        #用于存储状态的历史记录（用于计时）
        self.stand_start_time = 0
        self.sit_start_time = 0
        self.walk_start_time = 0
        
        # 新增视频文件相关变量
        self.video_path = None
        self.is_video_mode = False
        self.video_cap = None
        
        super().__init__()
        self.initUI()
        self.init_counter_updater()
    def initUI(self):
        # 设置窗口基本属性
        self.setWindowTitle('摄像头监控与计时界面')
        self.setGeometry(100, 100, 1080, 720)
        
        # 设置淡蓝色背景
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(200, 230, 255))
        self.setPalette(palette)
        
        # 创建主布局
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # 左侧控制面板（占1/3宽度）
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setFixedWidth(360)

        # 设置左侧面板样式 - 所有文字为黑色
        left_panel.setStyleSheet("""
            QLabel, QPushButton {
                color: black;
                font-size: 14px;
                font-weight: bold;
            }
            QLabel {
                font-size: 16px;
                margin: 5px;
            }
            QPushButton {
                background-color: #e0e0e0;
                border: 1px solid #a0a0a0;
                border-radius: 5px;
                padding: 8px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
            QPushButton:pressed {
                background-color: #c0c0c0;
            }
        """)
        
        # 摄像头控制按钮
        self.open_camera_btn = QPushButton('等待3s后打开摄像头')
        self.open_camera_btn.clicked.connect(self.open_camera)
        self.close_camera_btn = QPushButton('关闭摄像头')
        self.close_camera_btn.clicked.connect(self.close_camera)
        
        # 新增视频文件上传按钮
        self.upload_video_btn = QPushButton('上传本地视频文件')
        self.upload_video_btn.clicked.connect(self.upload_video)
        self.play_video_btn = QPushButton('播放视频文件')
        self.play_video_btn.clicked.connect(self.play_video)
        self.play_video_btn.setEnabled(False)  # 初始状态禁用
        
        # 状态显示标签
        self.status_label = QLabel('当前状态: 待机')
        self.left_hand_label = QLabel('举左手计数: 0')
        self.right_hand_label = QLabel('举右手计数: 0')
        self.stand_label = QLabel('站立时长计时: 00:00')
        self.sit_label = QLabel('坐下时长计时: 00:00')
        self.walk_label = QLabel('行走计时: 00:00')
        
        # 计时控制按钮
        self.start_all_btn = QPushButton('开始所有计时计数')
        self.start_all_btn.clicked.connect(self.start_all)
        self.stop_all_btn = QPushButton('关闭计时计数')
        self.stop_all_btn.clicked.connect(self.stop_all)
        self.clear_all_btn = QPushButton('清空所有计时计数')
        self.clear_all_btn.clicked.connect(self.clear_all)
        
        # 添加控件到左侧布局
        left_layout.addWidget(self.open_camera_btn)
        left_layout.addWidget(self.close_camera_btn)
        left_layout.addSpacing(10)
        left_layout.addWidget(self.upload_video_btn)
        left_layout.addWidget(self.play_video_btn)
        left_layout.addSpacing(20)
        left_layout.addWidget(self.status_label)
        left_layout.addWidget(self.left_hand_label)
        left_layout.addWidget(self.right_hand_label)
        left_layout.addWidget(self.stand_label)
        left_layout.addWidget(self.sit_label)
        left_layout.addWidget(self.walk_label)
        left_layout.addSpacing(20)
        left_layout.addWidget(self.start_all_btn)
        left_layout.addWidget(self.stop_all_btn)
        left_layout.addWidget(self.clear_all_btn)
        left_layout.addStretch()
        
        left_panel.setLayout(left_layout)
        
        # 右侧摄像头显示区域（占2/3宽度）
        self.camera_display = QFrame()
        self.camera_display.setFixedSize(720, 700)
        self.camera_display.setStyleSheet("background-color: black;")
        
        # 添加左右面板到主布局
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.camera_display)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # 初始化计时器和计数器
        self.init_timers_and_counters()
        
    def init_timers_and_counters(self):
        # 初始化各种计时器和计数器
        global L_handcounts, R_handcounts
        self.left_hand_count = L_handcounts
        self.right_hand_count = R_handcounts
        self.stand_time = 0
        self.sit_time = 0
        self.walk_time = 0
        self.reference_image = []
        # 初始化开始时间标记
        self.stand_started = False
        self.sit_started = False
        self.walk_started = False
        
        # 计时器
        self.stand_timer = QTimer()
        self.sit_timer = QTimer()
        self.walk_timer = QTimer()
    def upload_video(self):
        """上传本地视频文件"""
        print("按下：上传本地视频文件")
        # 支持多种视频格式
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "选择视频文件",
            "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v *.3gp *.mpg *.mpeg *.ts *.mts *.m2ts);;所有文件 (*)"
        )
        
        if file_path:
            self.video_path = file_path
            self.play_video_btn.setEnabled(True)
            print(f"已选择视频文件: {file_path}")
            # 显示文件名
            filename = os.path.basename(file_path)
            self.upload_video_btn.setText(f'已选择: {filename[:20]}...' if len(filename) > 20 else f'已选择: {filename}')
        else:
            print("未选择视频文件")
            
    def play_video(self):
        """播放视频文件"""
        print("按下：播放视频文件")
        if not self.video_path:
            return
            
        try:
            # 关闭摄像头（如果正在运行）
            self.close_camera()
            
            # 确保摄像头线程完全停止
            if hasattr(self, 'video_thread') and self.video_thread.is_alive():
                print("等待视频线程完全停止...")
                self.video_thread.join(timeout=2)  # 等待最多2秒
            
            # 设置视频模式
            self.is_video_mode = True
            self.update_status_label("视频播放模式")
            
            # 启动视频处理线程
            self.video_event = threading.Event()
            self.video_thread = threading.Thread(target=self.process_video, args=(self.video_event,))
            self.video_thread.start()
            
            # 启动定时器，用于更新视频画面
            self.camera_timer = QTimer()
            self.camera_timer.timeout.connect(self.update_frame)
            self.camera_timer.start(30)  # 约33fps
            
            print(f"开始播放视频: {self.video_path}")
            
        except Exception as e:
            print(f"播放视频时出错: {e}")
            
    def process_video(self, event):
        """处理视频文件的线程函数"""
        global frameyplopose, global_stand_status, global_sit_status, global_walk_status, status_lock, L_feet, R_feet, L_feetxy, R_feetxy
        
        try:
            # 打开视频文件
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print(f"无法打开视频文件: {self.video_path}")
                return
                
            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"视频信息 - FPS: {fps}, 总帧数: {total_frames}")
            
            # 加载YOLO模型
            model = YOLO('yolov8n-pose.pt')
            
            frame_count = 0
            while cap.isOpened() and not event.is_set():
                success, frame = cap.read()
                if not success:
                    print("视频播放完毕")
                    break
                    
                frame_count += 1
                
                # 使用相同的YOLO处理逻辑
                message = ""
                R_smleg = 0
                L_smleg = 0
                R_bigleg = 0
                L_bigleg = 0
                R_leg = 0
                L_leg = 0
                up_Rleg = False
                up_Lleg = False
                Rleg = False
                Lleg = False
                stand_condition = False
                sit_condition = False
                walk_condition = False
                
                results = model(frame, save=False, conf=0.5)
                
                if len(results[0].boxes) < 1:
                    # 如果没有检测到人，直接显示原帧
                    frameyplopose = frame
                    continue
                    
                # 处理检测到的人
                boxes = results[0].boxes[0]
                keypoints = results[0].keypoints[0]
                x, y = boxes.xyxy.cpu().numpy()[0][0], boxes.xyxy.cpu().numpy()[0][1]
                key = np.array(keypoints.xy.cpu().numpy())
                person = np.sum(key)
                
                if person > 0:
                    KEY = [False] * 17
                    for i in range(len(key[0])):
                        if key[0][i][1] + key[0][i][0] == 0:
                            KEY[i] = False
                        else:
                            KEY[i] = True
                            
                    # 计算腿部长度
                    if KEY[12] and KEY[14]:
                        R_bigleg = matrix_distance(key[0][14], key[0][12])
                    if KEY[11] and KEY[13]:
                        L_bigleg = matrix_distance(key[0][11], key[0][13])
                    if KEY[16] and KEY[14]:
                        R_smleg = matrix_distance(key[0][14], key[0][16])
                    if KEY[15] and KEY[13]:
                        L_smleg = matrix_distance(key[0][15], key[0][13])
                    if KEY[16] and KEY[12]:
                        Rleg = True
                        R_leg = matrix_distance(key[0][16], key[0][12])
                    if KEY[13] and KEY[11]:
                        Lleg = True
                        L_leg = matrix_distance(key[0][11], key[0][13])
                        
                    # 判断腿部弯曲
                    if (R_leg < (R_smleg + R_bigleg) * 0.92 and R_leg * R_smleg * R_bigleg != 0) or R_smleg > R_bigleg:
                        up_Rleg = True
                    if (L_leg < (L_smleg + L_bigleg) * 0.92 and L_leg * L_smleg * L_bigleg != 0) or L_smleg > L_bigleg:
                        up_Lleg = True
                        
                    # 判断举手
                    if KEY[9] and KEY[6] and (key[0][9][1] < key[0][6][1]):
                        message = message + "L_hand up "
                    if KEY[10] and KEY[5] and (key[0][10][1] < key[0][5][1]):
                        message = message + "R_hand up "
                        
                    # 更新脚部位置
                    if KEY[15] and KEY[16]:
                        L_feetxy = [int(key[0][15][0]), int(key[0][15][1])]
                        R_feetxy = [int(key[0][16][0]), int(key[0][16][1])]
                    else:
                        L_feetxy = [0, 0]
                        R_feetxy = [0, 0]
                        
                    # 行走检测逻辑 - 根据视频分辨率调整阈值
                    frame_height, frame_width = frame.shape[:2]
                    # 动态调整阈值，基于视频分辨率
                    base_threshold = 10
                    scale_factor = min(frame_width, frame_height) / 720  # 基于720p作为基准
                    lengTH = int(base_threshold * scale_factor)

                    if len(L_feet) ==3 and len(R_feet) ==3 and KEY[15] and KEY[16]:
                        # 计算左脚移动距离
                        left_move1 = matrix_distance(L_feet[0], L_feet[1])
                        left_move2 = matrix_distance(L_feet[1], L_feet[2])
                        # 计算右脚移动距离
                        right_move1 = matrix_distance(R_feet[0], R_feet[1])
                        right_move2 = matrix_distance(R_feet[1], R_feet[2])
                        
                        # 判断是否为行走（连续移动）
                        if ((left_move1 > lengTH and left_move2 > lengTH) or 
                            (right_move1 > lengTH and right_move2 > lengTH)):
                            walk_condition=True
                            sit_condition=False
                            stand_condition=False
                            message=message+"walk"
                            print(f"检测到行走 - 左脚移动: {left_move1:.1f}, {left_move2:.1f}, 右脚移动: {right_move1:.1f}, {right_move2:.1f}, 阈值: {lengTH}")
                        
                    # 判断姿态（只有在没有行走时才判断）
                    if not walk_condition:
                        if Lleg and Rleg : #左右腿是否成功检测
                            if up_Lleg and up_Rleg :#数学逻辑判断是否是坐姿（两腿弯曲）
                                sit_condition=True
                                message=""
                                message=message+"sit "
                            else :
                                stand_condition=True
                                message=""
                                message=message+"stand "#print("stand")
                        else :
                            sit_condition=True
                            message=message+"no leg sit "#print("no leg sit")
                        
                    # 更新全局状态
                    with status_lock:
                        global_stand_status = stand_condition
                        global_sit_status = sit_condition
                        global_walk_status = walk_condition
                        
                # 绘制结果
                annotated_frame = results[0][0].plot()
                font = cv2.FONT_HERSHEY_SIMPLEX
                annotated_frame = cv2.putText(annotated_frame, message, (int(x) + 10, int(y) + 30), font, 1, (0, 0, 255), 2)
                frameyplopose = annotated_frame
                
                # 控制播放速度
                time.sleep(1/fps)
                
            cap.release()
            print("视频处理完成")
            
        except Exception as e:
            print(f"处理视频时出错: {e}")
        finally:
            self.is_video_mode = False
            self.update_status_label("待机")

    # 按钮事件处理函数
    def open_camera(self):
        print("按下：打开摄像头")
        # 关闭视频播放（如果正在运行）
        if hasattr(self, 'video_event'):
            self.video_event.set()
            self.is_video_mode = False
            
        # 确保视频线程完全停止
        if hasattr(self, 'video_thread') and self.video_thread.is_alive():
            print("等待视频线程完全停止...")
            self.video_thread.join(timeout=2)  # 等待最多2秒
            
        try:
            # 等待3秒后启动摄像头
            print("等待3秒后启动摄像头...")
            
            # 启动摄像头线程（如果还没有启动）
            if hasattr(self, 'camera_thread') and not self.camera_thread.is_alive():
                self.camera_event.clear()  # 清除停止标志
                self.camera_thread.start()
                self.timestamp_thread.start()
                print("摄像头线程已启动")
                self.update_status_label("摄像头模式")
            
            # 替换为你的图片路径
            self.reference_image = frameyplopose
            print(self.reference_image)
            if self.reference_image is None:
                print(f"无法加载图片")
            else:
                print(f"成功加载图片")
                # 可以在这里对图片进行预处理
                self.reference_image = cv2.resize(self.reference_image, (720, 700))
                
        except Exception as e:
            print(f"加载图片时出错: {e}")
        
        # 启动定时器，用于更新摄像头画面
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_frame)
        self.camera_timer.start(30)  # 约33fps
        
    def update_frame(self):
        global frameyplopose

        # 如果图像未初始化，则跳过本帧
        if not isinstance(frameyplopose, np.ndarray) or frameyplopose.ndim != 3:
            return
        # 处理帧并显示
        print("---",frameyplopose)
        frame = cv2.resize(frameyplopose, (720, 700))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转换为QImage并显示在界面上
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # 在摄像头显示区域创建一个QLabel来显示图像
        if not hasattr(self, 'camera_label'):
            self.camera_label = QLabel()
            self.camera_display.layout = QVBoxLayout(self.camera_display)
            self.camera_display.layout.addWidget(self.camera_label)

        self.camera_label.setPixmap(pixmap)

    def close_camera(self):
        print("按下：关闭摄像头")

        # 停止视频播放（如果正在运行）
        if hasattr(self, 'video_event'):
            self.video_event.set()
            self.is_video_mode = False

        # 停止摄像头线程（如果正在运行）
        if hasattr(self, 'camera_event'):
            self.camera_event.set()
            print("摄像头线程已停止")

        # 停止定时器
        if hasattr(self, 'camera_timer'):
            self.camera_timer.stop()

        # 清空显示区域
        if hasattr(self, 'camera_label'):
            self.camera_label.clear()
            
        # 更新状态
        self.update_status_label("待机")

    def start_all(self):
        print("按下：开始所有计时")
        self.TimeBool =True
        self.init_counter_updater()
        self.update_display()

    def stop_all(self):
        print("按下：关闭计时")
        self.TimeBool =False
        self.sit_label.setText(f'坐下时长计时关闭')


    def clear_all(self):
        print("按下：清空所有计时计数")
        global L_Hand, L_handcounts, R_Hand, R_handcounts, sit_Time, stand_Time, walk_Time
        L_handcounts = 1
        R_handcounts = 1

        L_Hand = 0
        R_Hand = 0
        sit_Time = 0
        stand_Time = 0
        walk_Time = 0
        self.left_hand_count = 0
        self.right_hand_count = 0
        self.stand_time = 0
        self.sit_time = 0
        self.walk_time = 0

        # 重置开始时间标记
        self.stand_started = False
        self.sit_started = False
        self.walk_started = False
        self.update_display()

    def init_counter_updater(self):
        self.counter_timer = QTimer()
        self.counter_timer.timeout.connect(self.update_counters)
        self.counter_timer.start(200)  # 每0.2秒更新一次

    # 格式化时间显示 (秒 -> MM:SS)
    def format_time(self, seconds):
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:.1f}:{seconds:.1f}"

    # 更新所有显示

    def update_display(self):
        self.left_hand_label.setText(f'举左手计数: {self.left_hand_count}')
        self.right_hand_label.setText(f'举右手计数: {self.right_hand_count}')
        self.stand_label.setText(f'站立时长计时: {self.format_time(self.stand_time)}')
        self.sit_label.setText(f'坐下时长计时: {self.format_time(self.sit_time)}')
        self.walk_label.setText(f'行走计时: {self.format_time(self.walk_time)}')

    # 实际应从共享变量获取
    def get_stand_status(self):
        global global_stand_status
        return global_stand_status
    def get_sit_status(self):
        global global_sit_status
        return global_sit_status
    def get_walk_status(self):
        global global_walk_status
        return global_walk_status
    def get_left_hand_count(self):
        # 从共享变量获取左手计数
        global L_Hand
        return L_Hand
    def get_right_hand_count(self):
        # 从共享变量获取右手计数
        global R_Hand
        return R_Hand
    def get_stand_time(self):
        global stand_Time
        return stand_Time
    def get_sit_time(self):
        global sit_Time
        return sit_Time
    def get_walk_time(self):
        global walk_Time
        return walk_Time

    def update_counters(self):
        # 从共享变量获取最新状态
        global L_Hand, R_Hand
        if self.TimeBool:
            self.left_hand_count = L_Hand  # 左手计数
            self.right_hand_count = R_Hand  # 右手计数

        # 获取姿态状态（需要从线程安全的变量获取）
        self.stand_status = self.get_stand_status()
        self.sit_status = self.get_sit_status()
        self.walk_status = self.get_walk_status()

        # 更新计时
        current_time = time.time()
        if self.stand_status and self.TimeBool:
            if not hasattr(self, 'stand_started') or not self.stand_started:
                self.stand_start_time = current_time
                self.stand_started = True
            self.stand_time = self.stand_time + (current_time - self.stand_start_time)
            self.stand_start_time = current_time  # 重置开始时间，避免重复计算
        else:
            if hasattr(self, 'stand_started') and self.stand_started:
                self.stand_time = self.stand_time + (current_time - self.stand_start_time)
                self.stand_started = False

        # 更新坐姿计时
        if self.sit_status and self.TimeBool:
            if not hasattr(self, 'sit_started') or not self.sit_started:
                self.sit_start_time = current_time
                self.sit_started = True
            self.sit_time = self.sit_time + (current_time - self.sit_start_time)
            self.sit_start_time = current_time  # 重置开始时间
        else:
            if hasattr(self, 'sit_started') and self.sit_started:
                self.sit_time = self.sit_time + (current_time - self.sit_start_time)
                self.sit_started = False

        # 更新行走计时
        if self.walk_status and self.TimeBool:
            if not hasattr(self, 'walk_started') or not self.walk_started:
                self.walk_start_time = current_time
                self.walk_started = True
            self.walk_time = self.walk_time + (current_time - self.walk_start_time)
            self.walk_start_time = current_time  # 重置开始时间
        else:
            if hasattr(self, 'walk_started') and self.walk_started:
                self.walk_time = self.walk_time + (current_time - self.walk_start_time)
                self.walk_started = False

        # 更新显示
        self.update_display()

    def update_status_label(self, status):
        """更新状态标签"""
        self.status_label.setText(f'当前状态: {status}')

##KEY[0     1       2       3       4
## [“鼻子”、“左眼”、“右眼”、“左耳”、“右耳”、
#   5         6        7        8       9       10
#《左肩》、《右肩》、《左肘》、《右肘》、“左腕”、《右腕》、
##[  11      12         13      14          15        16  ]
##、《左髋》、《右髋》、“左膝”、《右膝》、《左脚踝》、《右脚踝》]
def matrix_distance(point_a, point_b):
    x1, y1 = point_a
    x2, y2 = point_b
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
#统计布尔数量
def compare_boolean_matrix(matrix):
    # 统计True和False的数量
    true_count = matrix.count(True)
    false_count = matrix.count(False)

    # 判断是否有一个元素与其他不同
    if true_count == 1 or false_count == 1:
        return 1
    else:
        return 0

def print_timestamp(event):
    global L_feet, R_feet, L_feetxy, R_feetxy, L_hand, R_hand, L_handup, R_handup, L_handcounts, R_handcounts
    """后台线程函数，用于每0.2秒记录一次当前所有状态"""
    while not event.is_set():

        if len(L_feet) < 3 and len(R_feet) < 3 : #队列记录3次左右脚位置存放在脚统计中
            L_feet.append(L_feetxy)
            R_feet.append(R_feetxy)
        else :
            L_feet.pop(0)
            R_feet.pop(0)
            L_feet.append(L_feetxy)
            R_feet.append(R_feetxy)

        if len(L_hand) < 2: #队列记录2次L_handup左手举放状态在L_hand中
            L_hand.append(L_handup)
        else :
            L_hand.pop(0)
            L_hand.append(L_handup)
        if len(R_hand) < 2: #队列记录2次R_handup右手举放状态在R_hand中
            R_hand.append(R_handup)
        else :
            R_hand.pop(0)
            R_hand.append(R_handup)
        #累加计数
        L_handcounts+=compare_boolean_matrix(L_hand)
        R_handcounts+=compare_boolean_matrix(R_hand)
        print("-------------------------------------------")
        print(L_handcounts)
        print(R_handcounts)
        print("-------------------------------------------")
        #状态锁设置walk、sit、stand状态
        with status_lock:
            current_stand = global_stand_status
            current_sit = global_sit_status
            current_walk = global_walk_status
        time.sleep(0.2)  # 延迟0.2秒（每0.2秒记录一次以上内容）


def yolo(event):
    model = YOLO('yolov8n-pose.pt')  # 加载预训练模型
    video_path = 0
    cap = cv2.VideoCapture(video_path)
    global L_feet, R_feet, L_feetxy, R_feetxy, frameyplopose, L_handup, R_handup, L_handcounts, R_handcounts, L_Hand, R_Hand, global_stand_status, global_sit_status, global_walk_status, status_lock
    hip=True
    ankle=True
    KEY=[False] * 17 #初始化17个关键点为False

    while cap.isOpened() and not event.is_set():
        # Read a frame from the video
        success, frame = cap.read()
        message="" #图片中显示姿态字符串
        R_smleg=0
        L_smleg=0
        R_bigleg=0
        L_bigleg=0
        R_leg=0
        L_leg=0
        up_Rleg=False
        up_Lleg=False
        Rleg=False
        Lleg=False
        stand_condition=False
        sit_condition=False
        walk_condition=False
        if success:

            results = model(frame,save=False,conf=0.6)

            if len(results[0].boxes)<1:
                continue
            #results[0].[0]#置信度最大的那个人
            boxes = results[0].boxes[0]  # Boxes object for bbox outputs
            keypoints = results[0].keypoints[0]  # Keypoints object for pose outputs
            x,y=boxes.xyxy.cpu().numpy()[0][0],boxes.xyxy.cpu().numpy()[0][1]
            key=np.array(keypoints.xy.cpu().numpy())
            person= np.sum(key) #至少检测到一个点
            if person>0:
                #print(key)
                for i in range(len(key[0])):
                    if key[0][i][1]+key[0][i][0]==0:
                        KEY[i]=False
                    else :KEY[i]=True
                print(KEY)  #点位数组
                #右大腿长R_bigleg
                if KEY[12] and KEY[14]:
                    R_bigleg=matrix_distance(key[0][14],key[0][12])
                #左大腿长L_bigleg
                if KEY[11] and KEY[13]:
                    L_bigleg=matrix_distance(key[0][11],key[0][13])
                #右小腿长R_smleg
                if KEY[16] and KEY[14]:
                    R_smleg=matrix_distance(key[0][14],key[0][16])
                #左小腿长L_smleg
                if KEY[15] and KEY[13]:
                    L_smleg=matrix_distance(key[0][15],key[0][13])
                #整右腿长
                if KEY[16] and KEY[12]:
                    Rleg=True
                    R_leg=matrix_distance(key[0][16],key[0][12])
                #整左腿长
                if KEY[13] and KEY[11]:
                    Lleg=True
                    L_leg=matrix_distance(key[0][11],key[0][13])
                #print(R_leg)
                #print(R_smleg)
                #print(R_bigleg)
                if (R_leg<(R_smleg+R_bigleg)*0.92 and R_leg*R_smleg*R_bigleg!=0 )or R_smleg>R_bigleg: # 数学逻辑判断是否弯曲右腿
                    up_Rleg=True
                    print("up Rleg")
                if (L_leg<(L_smleg+L_bigleg)*0.92 and L_leg*L_smleg*L_bigleg!=0 )or L_smleg>L_bigleg: # 数学逻辑判断是否弯曲左腿
                    up_Lleg=True
                    print("up Lleg")
                if KEY[9] and KEY[6] and (key[0][9][1]<key[0][6][1]) :
                    # 数学逻辑判断是否举起左手（左手点位高于左肩）
                    message=message+"L_hand up "
                    L_handup=True
                    L_Hand=int(L_handcounts*0.5) #一举一放视为一次完整的举手
                    message=message+str(L_Hand)
                else :L_handup=False
                if KEY[10] and KEY[5] and (key[0][10][1]<key[0][5][1]):
                     # 数学逻辑判断是否举起右手（右手点位高于右肩）
                    message=message+"R_hand up "
                    R_handup=True
                    R_Hand=int(R_handcounts*0.5) #一举一放视为一次完整的举手
                    message=message+str(R_Hand)
                else :R_handup=False
                if Lleg and Rleg : #左右腿是否成功检测
                    if up_Lleg and up_Rleg :#数学逻辑判断是否是坐姿（两腿弯曲）
                        if  walk_condition is False: #如果没有行走
                            sit_condition=True
                            message=""
                            message=message+"sit "
                    else :
                        if  walk_condition is False: #如果没有行走
                            stand_condition=True
                            message=""
                            message=message+"stand "#print("stand")
                else :

                    sit_condition=True
                    message=message+"no leg sit "#print("no leg sit")

                if KEY[15] and KEY[16]:
                    L_feetxy=([int(key[0][15][0]),int(key[0][15][1])])
                    R_feetxy=([int(key[0][16][0]),int(key[0][16][1])])
                else:
                    L_feetxy=[0,0]
                    R_feetxy=[0,0]

                # 行走检测逻辑 - 根据视频分辨率调整阈值
                frame_height, frame_width = frame.shape[:2]
                # 动态调整阈值，基于视频分辨率
                base_threshold = 5
                scale_factor = min(frame_width, frame_height) / 720  # 基于720p作为基准
                lengTH = int(base_threshold * scale_factor)

                if len(L_feet) ==3 and len(R_feet) ==3 and KEY[15] and KEY[16]:
                    # 计算左脚移动距离
                    left_move1 = matrix_distance(L_feet[0], L_feet[1])
                    left_move2 = matrix_distance(L_feet[1], L_feet[2])
                    # 计算右脚移动距离
                    right_move1 = matrix_distance(R_feet[0], R_feet[1])
                    right_move2 = matrix_distance(R_feet[1], R_feet[2])
                    
                    # 判断是否为行走（连续移动）
                    if ((left_move1 > lengTH and left_move2 > lengTH) or 
                        (right_move1 > lengTH and right_move2 > lengTH)):
                        walk_condition=True
                        sit_condition=False
                        stand_condition=False
                        message=message+"walk"
                        print(f"检测到行走 - 左脚移动: {left_move1:.1f}, {left_move2:.1f}, 右脚移动: {right_move1:.1f}, {right_move2:.1f}, 阈值: {lengTH}")
                        
                # 判断姿态（只有在没有行走时才判断）
                if not walk_condition:
                    if Lleg and Rleg : #左右腿是否成功检测
                        if up_Lleg and up_Rleg :#数学逻辑判断是否是坐姿（两腿弯曲）
                            sit_condition=True
                            message=""
                            message=message+"sit "
                        else :
                            stand_condition=True
                            message=""
                            message=message+"stand "#print("stand")
                    else :
                        sit_condition=True
                        message=message+"no leg sit "#print("no leg sit")

                # 更新全局状态
                with status_lock:
                    global_stand_status = stand_condition
                    global_sit_status = sit_condition
                    global_walk_status = walk_condition
            else:print("no key")
            annotated_frame = results[0][0].plot()
            font=cv2.FONT_HERSHEY_SIMPLEX
            annotated_frame = cv2.putText(annotated_frame, message,(int(x)+10,int(y)+30), font, 1, (0,0,255), 2)
            frameyplopose = annotated_frame
        else:
            # Break the loop if the end of the video is reached
            break
    cap.release()

if __name__ == '__main__':
    #初始化定义
    frameyplopose = np.zeros((700, 720, 3), dtype=np.uint8)
    L_feet = []
    R_feet = []
    L_feetxy = [0, 0]
    R_feetxy = [0, 0]
    L_hand = []
    R_hand = []
    L_handup = False
    R_handup = False
    L_handcounts = 0
    R_handcounts = 0
    L_Hand = 0
    R_Hand = 0
    global_stand_status = False
    global_sit_status = False
    global_walk_status = False
    sit_Time = 0
    walk_Time = 0
    stand_Time = 0
    status_lock = threading.Lock()

    app = QApplication(sys.argv)
    window = CameraInterface()
    window.show()

    # 创建事件对象，但不立即启动线程
    event = threading.Event()
    
    # 创建线程对象，但不立即启动
    threadyolo = threading.Thread(target=yolo, args=(event,))
    thread = threading.Thread(target=print_timestamp, args=(event,))
    
    # 将线程对象传递给窗口，以便控制启动和停止
    window.camera_event = event
    window.camera_thread = threadyolo
    window.timestamp_thread = thread
    
    sys.exit(app.exec_())
    

    try:
        print(1)
        
    # 创建应用实例并显示窗口
    
    except KeyboardInterrupt:
        print("程序被手动终止")
        event.set()
        thread.join()
        threadyolo.join()
        print("终止")
    #yolo()"""
    

