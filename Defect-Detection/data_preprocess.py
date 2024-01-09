# import json
#
# # 从 JSON 文件加载数据
# with open('attacker.json', 'r') as file:
#     data = json.load(file)
#
# # 读取所有的 'y' 值
# #new_x_raw = [item['new_x_raw'] for item in data]
# for item in data:
#     new_x_raw = item['new_x_raw']
#     #print("所有的 'y' 值:", new_x_raw)
# print(len(data))




import json
# import json
#
# # # 读取合并后的文件
# with open('attacker_80_1.json', 'r') as file:
#     merged_data = json.load(file)
#
# # 统计数据条数
# num_data = len(merged_data)
#
# # 打印数据条数
# print("合并后的文件中共有 {} 条数据。".format(num_data))




# 创建空的列表用于存储合并后的数据
# merged_data = []
# #
# # # 逐个读取JSON文件
# # count=0
# file_names = ['attacker_60_1.json', 'attacker_60_2.json', 'attacker_60_3.json','attacker_60_4.json','attacker_60_5.json']  # 替换为实际的文件名列表
# for file_name in file_names:
#     with open(file_name, 'r') as file:
#         data = json.load(file)
#         # 检查数据是否已存在于合并数据中，避免重复添加
#         for item in data:
#             if item not in merged_data:
#                 merged_data.append(item)
#
#
#
# # 将合并后的数据存储为新的JSON文件
# with open('merged_data.json', 'w') as file:
#     json.dump(merged_data, file)


import json

# 读取五个json文件的数据
data_files = ['attacker_60_1.json', 'attacker_60_2.json', 'attacker_60_3.json', 'attacker_60_4.json', 'attacker_60_5.json']



import json
files = ['attacker_60_1.json', 'attacker_60_2.json', 'attacker_60_3.json', 'attacker_60_4.json', 'attacker_60_5.json']
data1 = []
data2 = []
data3 = []
data4 = []
data5 = []

#将new_x_raw转换成字符串
# with open('attacker_60_1.json', "r") as f:
#     file_data = json.load(f)
#     for item in file_data:
#             new_x_raw = item['new_x_raw']
#             batch = {"x": []}
#             raws = [new_x_raw]
#             batch['x'] = [" ".join(x) for x in raws]
#             row = {
#                 'new_x_raw':batch['x']
#             }
#             data1.append(row)
#
#
# with open('data1.json', 'w') as file:
#     json.dump(data1, file)
#

