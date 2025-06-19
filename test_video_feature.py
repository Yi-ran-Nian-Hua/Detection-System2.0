#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试视频上传功能的简单脚本
"""

import sys
import os
from PyQt5.QtWidgets import QApplication, QFileDialog

def test_file_dialog():
    """测试文件对话框功能"""
    app = QApplication(sys.argv)
    
    # 测试文件对话框
    file_dialog = QFileDialog()
    file_path, _ = file_dialog.getOpenFileName(
        None,
        "选择视频文件",
        "",
        "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v *.3gp *.mpg *.mpeg *.ts *.mts *.m2ts);;所有文件 (*)"
    )
    
    if file_path:
        print(f"选择的文件: {file_path}")
        filename = os.path.basename(file_path)
        print(f"文件名: {filename}")
    else:
        print("未选择文件")
    
    app.quit()

if __name__ == "__main__":
    test_file_dialog() 