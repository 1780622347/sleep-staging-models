"""
AAUWSS 数据集提取脚本
从 Aalborg University Wearable Sleep Study 数据集提取 PPG 和 ECG 数据
参照 extract_dreamt_data.py 的处理逻辑
"""

import os
import pickle
import numpy as np
import pandas as pd
import h5py
from scipy import signal
from scipy.signal import butter, filtfilt, resample
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 数据集路径
BASE_PATH = r"D:\工作记录\算法测试20260413\Data_Sets\sleep\Aalborg University Wearable Sleep Study (AAUWSS)"
OUTPUT_DIR = r"D:\工作记录\算法测试20260413\Data_Sets\sleep\AAUWSS_processed"

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 采样率配置
PPG_FS_ORIGINAL = 64  # AAUWSS PPG 原始采样率 64Hz
ECG_FS_ORIGINAL = 200  # AAUWSS ECG 原始采样率 200Hz
TARGET_FS = 34.13  # 目标采样率
EPOCH_DURATION = 30  # 30秒 epoch

# 标签映射 (4分类: Wake, Light, Deep, REM)
STAGE_MAP = {
    'Wake': 0,
    'N1': 1,  # Light sleep
    'N2': 1,  # Light sleep
    'N3': 2,  # Deep sleep
    'REM': 3
}

def butter_lowpass(cutoff, fs, order=4):
    """低通滤波器"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return b, a

def butter_bandpass(lowcut, highcut, fs, order=4):
    """带通滤波器"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def preprocess_ppg(ppg_signal, original_fs=PPG_FS_ORIGINAL, target_fs=TARGET_FS):
    """
    PPG 预处理流程:
    1. 8Hz 低通滤波
    2. 重采样到目标采样率
    3. 裁剪到 mean±3std
    4. Z-score 标准化
    """
    # 1. 低通滤波 (8Hz)
    b, a = butter_lowpass(8, original_fs, order=4)
    filtered = filtfilt(b, a, ppg_signal)
    
    # 2. 重采样到目标采样率
    target_samples = int(len(filtered) * target_fs / original_fs)
    resampled = resample(filtered, target_samples)
    
    # 3. 裁剪到 mean±3std
    mean_val = np.mean(resampled)
    std_val = np.std(resampled)
    lower_bound = mean_val - 3 * std_val
    upper_bound = mean_val + 3 * std_val
    clipped = np.clip(resampled, lower_bound, upper_bound)
    
    # 4. Z-score 标准化
    standardized = (clipped - mean_val) / std_val
    
    return standardized

def preprocess_ecg(ecg_signal, original_fs=ECG_FS_ORIGINAL, target_fs=TARGET_FS):
    """
    ECG 预处理流程:
    1. 0.5-40Hz 带通滤波
    2. 重采样到目标采样率
    3. 裁剪到 mean±3std
    4. Z-score 标准化
    """
    # 1. 带通滤波 (0.5-40Hz)
    b, a = butter_bandpass(0.5, 40, original_fs, order=4)
    filtered = filtfilt(b, a, ecg_signal)
    
    # 2. 重采样到目标采样率
    target_samples = int(len(filtered) * target_fs / original_fs)
    resampled = resample(filtered, target_samples)
    
    # 3. 裁剪到 mean±3std
    mean_val = np.mean(resampled)
    std_val = np.std(resampled)
    lower_bound = mean_val - 3 * std_val
    upper_bound = mean_val + 3 * std_val
    clipped = np.clip(resampled, lower_bound, upper_bound)
    
    # 4. Z-score 标准化
    standardized = (clipped - mean_val) / std_val
    
    return standardized

def load_subject_data(subject_num):
    """加载单个受试者的数据"""
    subject_id = f"subject_{subject_num:02d}"
    
    # 加载 PPG 数据
    ppg_path = os.path.join(BASE_PATH, "aligned_sleep_data_set", "ppg", f"{subject_id}_ppg.pkl")
    with open(ppg_path, 'rb') as f:
        ppg_df = pickle.load(f)
    
    # 加载 ECG 数据
    ecg_path = os.path.join(BASE_PATH, "aligned_sleep_data_set", "ecg", f"{subject_id}_ecg.pkl")
    with open(ecg_path, 'rb') as f:
        ecg_df = pickle.load(f)
    
    # 加载标注数据
    ann_path = os.path.join(BASE_PATH, "annotations", f"{subject_id}_manual_annotation.xlsx")
    ann_df = pd.read_excel(ann_path)
    
    return ppg_df, ecg_df, ann_df, subject_id

def extract_signals_from_df(df, signal_type='ppg'):
    """从 DataFrame 提取信号数据"""
    # 找到信号列（数字列）
    signal_cols = [col for col in df.columns if isinstance(col, (int, np.integer))]
    
    # 提取信号数据
    signals = df[signal_cols].values
    
    return signals

def process_subject(subject_num):
    """处理单个受试者的数据"""
    print(f"Processing subject_{subject_num:02d}...")
    
    # 加载原始数据
    ppg_df, ecg_df, ann_df, subject_id = load_subject_data(subject_num)
    
    # 提取原始信号
    ppg_signals = extract_signals_from_df(ppg_df, 'ppg')
    ecg_signals = extract_signals_from_df(ecg_df, 'ecg')
    
    # 获取标签
    stages = ann_df['Sleep Stage'].values
    
    # 验证数据长度一致性
    n_epochs = len(ann_df)
    if len(ppg_signals) != n_epochs or len(ecg_signals) != n_epochs:
        print(f"  Warning: Data length mismatch. PPG: {len(ppg_signals)}, ECG: {len(ecg_signals)}, Ann: {n_epochs}")
        # 取最小长度
        n_epochs = min(len(ppg_signals), len(ecg_signals), n_epochs)
        ppg_signals = ppg_signals[:n_epochs]
        ecg_signals = ecg_signals[:n_epochs]
        stages = stages[:n_epochs]
    
    print(f"  Total epochs: {n_epochs}")
    
    # 预处理信号
    processed_ppg = []
    processed_ecg = []
    labels = []
    
    for i in tqdm(range(n_epochs), desc=f"  Processing {subject_id}"):
        # 预处理 PPG
        ppg_epoch = preprocess_ppg(ppg_signals[i])
        processed_ppg.append(ppg_epoch)
        
        # 预处理 ECG
        ecg_epoch = preprocess_ecg(ecg_signals[i])
        processed_ecg.append(ecg_epoch)
        
        # 映射标签
        stage = stages[i]
        if stage in STAGE_MAP:
            labels.append(STAGE_MAP[stage])
        else:
            labels.append(-1)  # 未知标签
    
    processed_ppg = np.array(processed_ppg)
    processed_ecg = np.array(processed_ecg)
    labels = np.array(labels)
    
    # 统计信息
    print(f"  PPG shape: {processed_ppg.shape}")
    print(f"  ECG shape: {processed_ecg.shape}")
    print(f"  Labels distribution: {np.bincount(labels[labels >= 0])}")
    
    return processed_ppg, processed_ecg, labels, subject_id, n_epochs

def main():
    """主处理流程"""
    print("=" * 60)
    print("AAUWSS Data Extraction")
    print("=" * 60)
    
    all_ppg = []
    all_ecg = []
    all_labels = []
    subject_indices = []
    current_idx = 0
    
    # 处理所有受试者
    subject_list = list(range(1, 14))  # subject_01 到 subject_13
    
    for subject_num in subject_list:
        try:
            ppg, ecg, labels, subject_id, n_epochs = process_subject(subject_num)
            
            all_ppg.append(ppg)
            all_ecg.append(ecg)
            all_labels.append(labels)
            
            # 记录受试者索引
            subject_indices.append({
                'subject_id': subject_id,
                'start_idx': current_idx,
                'end_idx': current_idx + n_epochs,
                'n_epochs': n_epochs
            })
            current_idx += n_epochs
            
        except Exception as e:
            print(f"  Error processing subject_{subject_num:02d}: {e}")
            continue
    
    # 合并所有数据
    print("\n" + "=" * 60)
    print("Merging all subjects...")
    print("=" * 60)
    
    all_ppg = np.concatenate(all_ppg, axis=0)
    all_ecg = np.concatenate(all_ecg, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    print(f"Total PPG shape: {all_ppg.shape}")
    print(f"Total ECG shape: {all_ecg.shape}")
    print(f"Total labels: {len(all_labels)}")
    print(f"Labels distribution: {np.bincount(all_labels[all_labels >= 0])}")
    
    # 保存数据
    print("\n" + "=" * 60)
    print("Saving processed data...")
    print("=" * 60)
    
    # 保存 PPG + 标签
    ppg_output_path = os.path.join(OUTPUT_DIR, "aauwss_ppg_with_labels.h5")
    with h5py.File(ppg_output_path, 'w') as f:
        f.create_dataset('ppg', data=all_ppg, compression='gzip')
        f.create_dataset('labels', data=all_labels, compression='gzip')
    print(f"Saved: {ppg_output_path}")
    
    # 保存 ECG
    ecg_output_path = os.path.join(OUTPUT_DIR, "aauwss_ecg.h5")
    with h5py.File(ecg_output_path, 'w') as f:
        f.create_dataset('ecg', data=all_ecg, compression='gzip')
    print(f"Saved: {ecg_output_path}")
    
    # 保存受试者索引
    index_output_path = os.path.join(OUTPUT_DIR, "aauwss_subject_index.h5")
    with h5py.File(index_output_path, 'w') as f:
        for idx_info in subject_indices:
            grp = f.create_group(idx_info['subject_id'])
            grp.attrs['start_idx'] = idx_info['start_idx']
            grp.attrs['end_idx'] = idx_info['end_idx']
            grp.attrs['n_epochs'] = idx_info['n_epochs']
    print(f"Saved: {index_output_path}")
    
    print("\n" + "=" * 60)
    print("AAUWSS Data Extraction Complete!")
    print("=" * 60)
    
    return all_ppg, all_ecg, all_labels, subject_indices

if __name__ == "__main__":
    main()