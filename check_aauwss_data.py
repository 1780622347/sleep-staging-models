"""检查 AAUWSS 数据集结构 - 统计所有受试者的睡眠分期比例"""
import pickle
import numpy as np
import pandas as pd
import os

# 数据集路径
BASE_PATH = r"D:\工作记录\算法测试20260413\Data_Sets\sleep\Aalborg University Wearable Sleep Study (AAUWSS)"

# 统计所有受试者的睡眠分期分布
print("=" * 80)
print("AAUWSS 所有受试者睡眠分期类别统计")
print("=" * 80)

all_stats = []
total_stats = {'Wake': 0, 'N1': 0, 'N2': 0, 'N3': 0, 'REM': 0}

for subject_num in range(1, 14):  # subject_01 到 subject_13
    subject_id = f"subject_{subject_num:02d}"
    ann_path = os.path.join(BASE_PATH, "annotations", f"{subject_id}_manual_annotation.xlsx")
    
    try:
        ann_df = pd.read_excel(ann_path)
        
        # 统计各阶段数量
        stage_counts = ann_df['Sleep Stage'].value_counts()
        
        total_epochs = len(ann_df)
        stats = {
            'Subject': subject_id,
            'Total': total_epochs,
            'Wake': stage_counts.get('Wake', 0),
            'N1': stage_counts.get('N1', 0),
            'N2': stage_counts.get('N2', 0),
            'N3': stage_counts.get('N3', 0),
            'REM': stage_counts.get('REM', 0)
        }
        
        # 计算百分比
        for stage in ['Wake', 'N1', 'N2', 'N3', 'REM']:
            stats[f'{stage}%'] = stats[stage] / total_epochs * 100
            total_stats[stage] += stats[stage]
        
        all_stats.append(stats)
        
    except Exception as e:
        print(f"Error loading {subject_id}: {e}")

# 创建统计表格
stats_df = pd.DataFrame(all_stats)

# 打印详细统计
print("\n各受试者睡眠分期数量统计:")
print("-" * 80)
print(f"{'Subject':<12} {'Total':>7} {'Wake':>6} {'N1':>6} {'N2':>6} {'N3':>6} {'REM':>6}")
print("-" * 80)
for _, row in stats_df.iterrows():
    print(f"{row['Subject']:<12} {row['Total']:>7} {row['Wake']:>6} {row['N1']:>6} {row['N2']:>6} {row['N3']:>6} {row['REM']:>6}")
print("-" * 80)
total_epochs = sum([s['Total'] for s in all_stats])
print(f"{'Total':<12} {total_epochs:>7} {total_stats['Wake']:>6} {total_stats['N1']:>6} {total_stats['N2']:>6} {total_stats['N3']:>6} {total_stats['REM']:>6}")

print("\n\n各受试者睡眠分期百分比统计:")
print("-" * 80)
print(f"{'Subject':<12} {'Wake%':>8} {'N1%':>8} {'N2%':>8} {'N3%':>8} {'REM%':>8}")
print("-" * 80)
for _, row in stats_df.iterrows():
    print(f"{row['Subject']:<12} {row['Wake%']:>7.1f}% {row['N1%']:>7.1f}% {row['N2%']:>7.1f}% {row['N3%']:>7.1f}% {row['REM%']:>7.1f}%")
print("-" * 80)
print(f"{'Overall%':<12} {total_stats['Wake']/total_epochs*100:>7.1f}% {total_stats['N1']/total_epochs*100:>7.1f}% {total_stats['N2']/total_epochs*100:>7.1f}% {total_stats['N3']/total_epochs*100:>7.1f}% {total_stats['REM']/total_epochs*100:>7.1f}%")

print("\n\n总体睡眠分期分布:")
print("-" * 40)
total_all = sum(total_stats.values())
for stage, count in total_stats.items():
    print(f"{stage:<8}: {count:>6} ({count/total_all*100:.1f}%)")
print("-" * 40)
print(f"{'Total':<8}: {total_all:>6}")
