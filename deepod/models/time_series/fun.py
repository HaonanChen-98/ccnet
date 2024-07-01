# 假设 'C:\\path_to_your_file\\data.txt' 是你的TXT文件的路径
# 请替换为实际的文件路径
input_filename = 'C:\\Users\\Administrator\\Desktop\\实验最终结果.txt'

# 假设 'C:\\path_to_your_file\\data.csv' 是你希望保存CSV文件的路径
# 请替换为实际的文件路径
output_filename = 'C:\\Users\\Administrator\\Desktop\\\实验结果2.csv'

# 用于存储读取的数据
lines = []

# 打开TXT文件并按行读取
try:
    with open(input_filename, 'r', encoding='utf-8') as file:
        for line in file:
            # 移除每行的首尾空白字符（包括换行符），然后按逗号分割
            processed_line = line.strip().split(',')
            # 将分割后的数据添加到列表中
            lines.append(processed_line)

    # 将数据写入CSV文件
    with open(output_filename, 'w', encoding='utf-8') as file:
        for line in lines:
            # 将列表转换为字符串，用逗号连接，并添加换行符
            file.write(','.join('"{}"'.format(item) for item in line) + '\n')

except FileNotFoundError:
    print(f"The file {input_filename} does not exist. Please check the path.")
except Exception as e:
    print(f"An error occurred: {e}")