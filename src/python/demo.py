import subprocess

# 检查剪贴板中的数据类型
def check_clipboard_type():
    # 调用pbpaste命令获取剪贴板内容，并设置输出格式为数据类型
    result = subprocess.run(['pbpaste', '-Prefer', 'txt'], capture_output=True, text=True)
    
    if result.returncode == 0 and result.stdout:
        # 如果返回码为0且stdout不为空，则剪贴板中为文本数据
        print('剪贴板中的内容是文本：', result.stdout)
    else:
        # 剪贴板中可能是图片数据
        print('剪贴板中的内容可能是图片。')

# 检查剪贴板类型并执行相应操作
check_clipboard_type()
