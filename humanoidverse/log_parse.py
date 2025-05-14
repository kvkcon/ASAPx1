from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import os

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

# 选择步骤0-10和51-60的数据
target_steps_early = list(range(0, 11))  # 0到10
target_steps_late = list(range(51, 61))  # 51到60
all_target_steps = target_steps_early + target_steps_late

# 先获取所有可用步骤
available_steps = set()
for tag_data in data.values():
    available_steps.update(tag_data['step'])

# 过滤掉不可用的步骤
filtered_steps = [step for step in all_target_steps if step in available_steps]
filtered_data = {'step': filtered_steps}

for tag in tags:
    step_to_value = {s: v for s, v in zip(data[tag]['step'], data[tag]['value'])}
    filtered_data[tag] = [step_to_value.get(s, None) for s in filtered_steps]

# 创建DataFrame
filtered_df = pd.DataFrame(filtered_data)

# 保存为CSV
filtered_df.to_csv("steps_0_10_and_51_60_metrics.csv", index=False)
print("已导出第0-10步和第51-60步的所有指标到 steps_0_10_and_51_60_metrics.csv")

# 打印任何缺失的步骤
missing_steps = set(all_target_steps) - available_steps
if missing_steps:
    print(f"警告: 以下步骤在日志中不存在: {sorted(list(missing_steps))}")
    print(f"可用的步骤范围: {min(available_steps)} - {max(available_steps)}")

# 按照Training Log格式输出为文本文件
with open("training_log_steps_0_10_and_51_60.txt", "w") as f:
    for step in all_target_steps:
        if step in available_steps:
            f.write(f"╭───────────────────────────────────────────────────────── Training Log ──────────────────────────────────────────────────────────╮\n")
            f.write(f"│                     Learning iteration {step}/1000000                                                                      │\n")
            f.write(f"│                                                                                                                                 │\n")
            
            # 计算这个步骤在filtered_steps中的索引
            idx = filtered_steps.index(step)
            
            # 对tag进行分类: 先显示主要指标，再显示环境指标，最后显示奖励指标
            main_metrics = []
            env_metrics = []
            reward_metrics = []
            
            for tag in tags:
                if tag.startswith("Env/"):
                    env_metrics.append(tag)
                elif tag.startswith("Mean episode rew_"):
                    reward_metrics.append(tag)
                else:
                    main_metrics.append(tag)
                    
            # 排序所有标签列表
            main_metrics.sort()
            env_metrics.sort()
            reward_metrics.sort()
            
            # 按顺序输出所有指标
            for tag in main_metrics + env_metrics + reward_metrics:
                value = filtered_data[tag][idx]
                if value is not None:
                    # 计算填充所需的空格数以达到格式对齐
                    metric_padding = 35  # 根据需要调整
                    value_str = f"{value:.4f}" if isinstance(value, float) else str(value)
                    spaces = " " * (130 - len(tag) - len(value_str) - 7)  # 7为其他固定字符的长度
                    
                    formatted_line = f"│ {tag:{metric_padding}}: {value_str}{spaces}│\n"
                    f.write(formatted_line)
            
            # 添加分隔线和总结信息（如果有的话）
            f.write(f"│ --------------------------------------------------------------------------------                                                │\n")
            
            # 寻找总时间步数等信息（如果有的话）
            total_timesteps_tag = next((t for t in tags if "timesteps" in t.lower()), None)
            if total_timesteps_tag and total_timesteps_tag in filtered_data:
                value = filtered_data[total_timesteps_tag][idx]
                if value is not None:
                    spaces = " " * (130 - len(total_timesteps_tag) - len(str(int(value))) - 7)
                    f.write(f"│ {total_timesteps_tag:{metric_padding}}: {int(value)}{spaces}│\n")
            
            # 结束框
            f.write(f"╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯\n\n")

print("已按照Training Log格式输出到 training_log_steps_0_10_and_51_60.txt")