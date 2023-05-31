import tkinter as tk
import pyperclip
import subprocess
import pytesseract
from PIL import ImageGrab
from AnswerFromOpenAI import *
from config import line


class ClipboardWatcher:
    
    def __init__(self):
        # 创建窗口
        self.root = tk.Tk()

        # 创建字符串变量并设置初始值
        self.clipboard_var = tk.StringVar()
        self.clipboard_var.set(pyperclip.paste())

        # 创建标签并绑定到字符串变量
        self.label = tk.Label(self.root, textvariable=self.clipboard_var, font=('Apple', 20), wraplength=500)

        # 将标签添加到窗口并显示
        self.label.pack()

        # 设置窗口置顶
        self.root.wm_attributes("-topmost", True)

        #tkinter设置自动换行
        self.label.pack(fill=tk.BOTH, expand=True)

        # 每秒钟更新标签
        self.root.after(1000, self.update_clipboard)

        # 运行GUI循环
        self.root.mainloop()

    def update_clipboard(self):
        # 获取剪贴板内容并更新字符串变量
        clipboard_content = pyperclip.paste()
        show_content = ''
        if self.check_clipboard_type():
            image = ImageGrab.grabclipboard()
            if image is not None:
                image = image.convert('L') #将图像转为灰度图像
                clipboard_content = pytesseract.image_to_string(image, lang='chi_sim')
                show_content = clipboard_content + line + AnswerFromOpenAI(clipboard_content)

        else:
            show_content = clipboard_content + line + AnswerFromOpenAI(clipboard_content)
        self.clipboard_var.set(show_content)

        # 调度下一次更新
        self.root.after(500, self.update_clipboard)

    def check_clipboard_type(self) -> bool:
        # 调用pbpaste命令获取剪贴板内容，并设置输出格式为数据类型
        result = subprocess.run(['pbpaste', '-Prefer', 'txt'], capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout:
            # 如果返回码为0且stdout不为空，则剪贴板中为文本数据
            return False
            print('剪贴板中的内容是文本：', result.stdout)
        else:
            # 剪贴板中可能是图片数据
            return True
            print('剪贴板中的内容可能是图片。')
    


if __name__ == '__main__':
    ClipboardWatcher()
