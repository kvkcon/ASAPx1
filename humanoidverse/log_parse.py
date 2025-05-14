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

# 只选择步骤50、51和52的数据
target_steps = [50, 51, 52]
filtered_data = {'step': target_steps}

for tag in tags:
    step_to_value = {s: v for s, v in zip(data[tag]['step'], data[tag]['value'])}
    filtered_data[tag] = [step_to_value.get(s, None) for s in target_steps]

# 创建DataFrame
filtered_df = pd.DataFrame(filtered_data)

# 保存为CSV
filtered_df.to_csv("steps_50_51_52_metrics.csv", index=False)
print("已导出第50、51和52步的所有指标到 steps_50_51_52_metrics.csv")

# 如果步骤50、51或52不在日志中，打印警告
available_steps = set()
for tag_data in data.values():
    available_steps.update(tag_data['step'])

missing_steps = set(target_steps) - available_steps
if missing_steps:
    print(f"警告: 步骤 {missing_steps} 在日志中不存在")
    print(f"可用的步骤范围: {min(available_steps)} - {max(available_steps)}")

# 在上面的代码之后添加

# 按照Training Log格式输出为文本文件
with open("training_log_steps_50_51_52.txt", "w") as f:
    for step in target_steps:
        if step in available_steps:
            f.write(f"╭───────────────────────────────────────────────────────── Training Log ──────────────────────────────────────────────────────────╮\n")
            f.write(f"│                     Learning iteration {step}/1000000                                                                      │\n")
            f.write(f"│                                                                                                                                 │\n")
            
            # 写入每个指标
            for tag in sorted(tags):
                idx = target_steps.index(step)
                value = filtered_data[tag][idx]
                if value is not None:
                    # 格式化输出，左对齐字段名，右对齐数值
                    formatted_line = f"│ {tag:<35}: {value:>7.4f}                                                                                     │\n"
                    f.write(formatted_line)
            
            f.write(f"╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯\n\n")

print("已按照Training Log格式输出到 training_log_steps_50_51_52.txt")