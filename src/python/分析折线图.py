import pandas as pd
import matplotlib.pyplot as plt

# 从多个CSV文件中读取数据
csv_files = ['auto.csv', 'way.csv', 'dlib.csv']  # 替换为您实际的CSV文件列表
data = []
name = {'auto.csv': 'dlib+adaptiveTrack', 'way.csv': 'dlib+Track', 'dlib.csv': 'dlib'}
for file in csv_files:
    df = pd.read_csv(file)
    data.append(df)
    df['label_column_name'] = name[file]  # 替换为包含标注的实际列名

# 创建折线图并添加标注
line_styles = ['-', '--', ':']  # 每条折线对应的线型
for idx, df in enumerate(data):
    x = df['count']
    y = df['fps']
    label = df['label_column_name'].iloc[0]  # 替换为包含标注的实际列名
    line_style = line_styles[idx % len(line_styles)]  # 循环使用线型列表中的样式

    plt.plot(x, y, label=label, linestyle=line_style)

    # 添加标注
    last_x = x.iloc[-1]
    last_y = y.iloc[-1]
    plt.annotate(label, (last_x, last_y), xytext=(10, -10), textcoords='offset points')

# 添加标签和标题
plt.xlabel('frame')
plt.ylabel('fps')
plt.title('fps of dlib')
plt.legend()  # 显示图例

# 显示图形
plt.show()