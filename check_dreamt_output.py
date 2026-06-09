import h5py
import os
import numpy as np

output_path = r'D:\工作记录\算法测试20260413\sleep_staging\sleep-staging-models\dreamt_processed'

# 检查生成的文件
files = os.listdir(output_path)
print('Generated files:')
for f in files:
    print(f'  {f}')

# 检查PPG文件内容
ppg_file = os.path.join(output_path, 'dreamt_ppg_with_labels.h5')
print('\nPPG file contents:')
with h5py.File(ppg_file, 'r') as f:
    print(f'  Keys: {list(f.keys())}')
    print(f'  PPG shape: {f["ppg"].shape}')
    print(f'  Labels shape: {f["labels"].shape}')
    labels = f['labels'][:]
    unique, counts = np.unique(labels, return_counts=True)
    print(f'  Labels distribution: {dict(zip(unique, counts))}')
    print(f'  Sampling rate: {f.attrs["sampling_rate"]}')
    print(f'  Total subjects: {f.attrs["total_subjects"]}')

# 检查ECG文件内容
ecg_file = os.path.join(output_path, 'dreamt_ecg.h5')
print('\nECG file contents:')
with h5py.File(ecg_file, 'r') as f:
    print(f'  Keys: {list(f.keys())}')
    print(f'  ECG shape: {f["ecg"].shape}')
    print(f'  Windows with real ECG: {f.attrs["windows_with_real_ecg"]}')

# 检查统计文件
stats_file = os.path.join(output_path, 'data_stats.txt')
print('\nStatistics:')
with open(stats_file, 'r', encoding='utf-8') as f:
    print(f.read())