import pandas as pd
import matplotlib.pyplot as plt

# 从多个CSV文件中读取数据
csv_files = ['auto.csv', 'way.csv', 'dlib.csv']  # 替换为您实际的CSV文件列表
data = []

for file in csv_files:
    df = pd.read_csv(file)
    data.append(df)

# 创建折线图
for df in data:
    x = df['count']
    y = df['flag']
    plt.scatter(x, y)

# 添加标签和标题
plt.xlabel('frame')
plt.ylabel('right flag')
plt.title('right rate of dlib')

# 显示图形
plt.show()