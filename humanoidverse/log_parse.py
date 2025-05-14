from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

# 指定events文件路径
event_file = "./events.out.tfevents.1746691631.michael-System-Product-Name.1009548.0"

# 加载事件文件
ea = event_accumulator.EventAccumulator(event_file)
ea.Reload()

# 获取所有可用的标签
tags = ea.Tags()['scalars']

# 创建数据字典
data = {}

# 遍历每个标签，提取数据
for tag in tags:
    events = ea.Scalars(tag)
    data[tag] = {
        'step': [event.step for event in events],
        'value': [event.value for event in events],
        'wall_time': [event.wall_time for event in events]
    }

# 转换为DataFrame并导出到CSV

# merge CSV：
all_steps = set()
for tag_data in data.values():
    all_steps.update(tag_data['step'])

all_steps = sorted(list(all_steps))
merged_data = {'step': all_steps}

for tag in tags:
    step_to_value = {s: v for s, v in zip(data[tag]['step'], data[tag]['value'])}
    merged_data[tag] = [step_to_value.get(s, None) for s in all_steps]

merged_df = pd.DataFrame(merged_data)
merged_df.to_csv("all_metrics.csv", index=False)
print("已导出所有指标到 all_metrics.csv")