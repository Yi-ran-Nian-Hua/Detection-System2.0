#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyQt5测试脚本
用于验证PyQt5是否正确安装和配置
"""

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('PyQt5测试窗口')
        self.setGeometry(100, 100, 400, 300)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建布局
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # 创建标签
        label = QLabel('PyQt5安装成功！\n这是一个测试窗口。')
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 16px; color: green;")
        layout.addWidget(label)
        
        # 添加系统信息
        info_label = QLabel(f'Python版本: {sys.version}\nPyQt5版本: 5.15.11')
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("font-size: 12px; color: blue;")
        layout.addWidget(info_label)

def main():
    print("启动PyQt5测试...")
    
    try:
        app = QApplication(sys.argv)
        print("QApplication创建成功")
        
        window = TestWindow()
        print("测试窗口创建成功")
        
        window.show()
        print("窗口显示成功")
        
        print("PyQt5测试完成，程序正常运行！")
        return app.exec_()
        
    except Exception as e:
        print(f"PyQt5测试失败: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 