"""
AAUWSS 数据集提取脚本
从 Aalborg University Wearable Sleep Study 数据集提取 PPG 和 ECG 数据
完全参照 extract_dreamt_data.py 的处理逻辑，确保生成一致的样本格式
每个受试者输出固定 (1200, 1024) 格式
"""

import os
import pickle
import numpy as np
import pandas as pd
import h5py
from scipy import signal
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

# 数据集路径
BASE_PATH = r"D:\工作记录\算法测试20260413\Data_Sets\sleep\Aalborg University Wearable Sleep Study (AAUWSS)"
OUTPUT_DIR = r"D:\工作记录\算法测试20260413\Data_Sets\sleep\AAUWSS_processed"

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 采样率配置 - 与DREAMT保持一致
PPG_FS_ORIGINAL = 64  # AAUWSS PPG 原始采样率 64Hz
ECG_FS_ORIGINAL = 200  # AAUWSS ECG 原始采样率 200Hz
TARGET_FS = 34.133333333  # 目标采样率 - 与DREAMT/MESA一致
WINDOW_DURATION = 30  # 30秒 epoch
SAMPLES_PER_WINDOW = 1024  # 每个窗口的采样点数 - 与DREAMT/MESA一致
WINDOWS_PER_SUBJECT = 1200  # 每个受试者的窗口数 - 与DREAMT/MESA一致
TARGET_LENGTH = WINDOWS_PER_SUBJECT * SAMPLES_PER_WINDOW  # 1,228,800

# 标签映射 (4分类: Wake, Light, Deep, REM)
STAGE_MAP = {
    'Wake': 0,
    'N1': 1,  # Light sleep
    'N2': 1,  # Light sleep
    'N3': 2,  # Deep sleep
    'REM': 3
}


def preprocess_ppg(ppg_signal, original_fs=PPG_FS_ORIGINAL, target_fs=TARGET_FS):
    """
    PPG 预处理流程 - 与DREAMT/MESA一致
    
    步骤:
    1. 8Hz低通滤波（Chebyshev II型，抗混叠）
    2. 重采样到34.13Hz（使用线性插值）
    3. 裁剪到mean±3std
    4. Z-score标准化
    
    Args:
        ppg_signal: 原始PPG信号
        original_fs: 原始采样率
        target_fs: 目标采样率
        
    Returns:
        处理后的PPG信号
    """
    # 处理NaN值
    ppg_signal = pd.Series(ppg_signal).fillna(method='ffill').fillna(method='bfill').values
    
    # 8Hz低通滤波（Chebyshev II型，与DREAMT/MESA一致）
    nyq = 0.5 * original_fs
    cutoff = 8 / nyq
    sos = signal.cheby2(N=8, rs=40, Wn=cutoff, btype='lowpass', output='sos')
    filtered_ppg = signal.sosfiltfilt(sos, ppg_signal)
    
    # 重采样到目标频率34.13Hz（使用线性插值，与DREAMT/MESA一致）
    duration = len(filtered_ppg) / original_fs
    n_samples = int(duration * target_fs)
    
    old_indices = np.linspace(0, len(filtered_ppg) - 1, len(filtered_ppg))
    new_indices = np.linspace(0, len(filtered_ppg) - 1, n_samples)
    downsampled_ppg = np.interp(new_indices, old_indices, filtered_ppg)
    
    # 裁剪到mean±3std
    mean = np.mean(downsampled_ppg)
    std = np.std(downsampled_ppg)
    clipped_ppg = np.clip(downsampled_ppg, mean - 3 * std, mean + 3 * std)
    
    # Z-score标准化
    wavppg = (clipped_ppg - np.mean(clipped_ppg)) / np.std(clipped_ppg)
    
    return wavppg


def preprocess_ecg(ecg_signal, original_fs=ECG_FS_ORIGINAL, target_fs=TARGET_FS):
    """
    ECG 预处理流程 - 与DREAMT/MESA一致
    
    步骤:
    1. 0.5-40Hz带通滤波
    2. 重采样到34.13Hz（使用线性插值）
    3. 裁剪到mean±3std
    4. Z-score标准化
    
    Args:
        ecg_signal: 原始ECG信号
        original_fs: 原始采样率
        target_fs: 目标采样率
        
    Returns:
        处理后的ECG信号
    """
    # 处理NaN值
    ecg_signal = pd.Series(ecg_signal).fillna(method='ffill').fillna(method='bfill').values
    
    # 0.5-40Hz带通滤波（与DREAMT/MESA一致）
    nyq = 0.5 * original_fs
    low = 0.5 / nyq
    high = 40.0 / nyq
    
    if high >= 1:
        high = 0.99
    
    b, a = butter(4, [low, high], btype='band')
    filtered_ecg = filtfilt(b, a, ecg_signal)
    
    # 重采样到目标频率34.13Hz（使用线性插值，与DREAMT/MESA一致）
    duration = len(filtered_ecg) / original_fs
    n_samples = int(duration * target_fs)
    
    old_indices = np.linspace(0, len(filtered_ecg) - 1, len(filtered_ecg))
    new_indices = np.linspace(0, len(filtered_ecg) - 1, n_samples)
    downsampled_ecg = np.interp(new_indices, old_indices, filtered_ecg)
    
    # 裁剪到mean±3std
    mean = np.mean(downsampled_ecg)
    std = np.std(downsampled_ecg)
    clipped_ecg = np.clip(downsampled_ecg, mean - 3 * std, mean + 3 * std)
    
    # Z-score标准化
    standardized_ecg = (clipped_ecg - np.mean(clipped_ecg)) / np.std(clipped_ecg)
    
    return standardized_ecg


def pad_or_truncate(signal, target_length, mode='truncate'):
    """
    将信号填充或截断到目标长度 - 与DREAMT一致
    
    Args:
        signal: 输入信号
        target_length: 目标长度
        mode: 'truncate' (截断) 或 'pad' (填充)
        
    Returns:
        处理后的信号
    """
    current_length = len(signal)
    
    if current_length >= target_length:
        # 截断
        return signal[:target_length], 'truncate', current_length - target_length
    else:
        # 填充（使用信号末尾值进行填充，避免边缘效应）
        padding_length = target_length - current_length
        # 使用信号的均值进行填充，保持信号特性
        padding = np.zeros(padding_length)
        return np.concatenate([signal, padding]), 'pad', padding_length


def pad_labels(labels, target_windows, original_windows):
    """
    处理标签的填充或截断
    
    Args:
        labels: 原始标签数组
        target_windows: 目标窗口数 (1200)
        original_windows: 原始窗口数
        
    Returns:
        处理后的标签数组
    """
    if original_windows >= target_windows:
        # 截断标签
        return labels[:target_windows], 'truncate', original_windows - target_windows
    else:
        # 填充标签（使用-1表示无效标签，训练时会忽略）
        padding_length = target_windows - original_windows
        padding = np.full(padding_length, -1)  # -1表示无效标签
        return np.concatenate([labels, padding]), 'pad', padding_length


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
    """从 DataFrame 提取信号数据并合并为连续信号"""
    # 找到信号列（数字列）
    signal_cols = [col for col in df.columns if isinstance(col, (int, np.integer))]
    
    # 提取信号数据并合并为连续信号
    signals = df[signal_cols].values
    
    # 合并所有epoch为连续信号
    continuous_signal = signals.flatten()
    
    return continuous_signal


def process_subject(subject_num):
    """
    处理单个受试者的数据 - 与DREAMT处理方式一致
    
    流程:
    1. 加载原始epoch数据
    2. 合并为连续信号
    3. 预处理（滤波、重采样、裁剪、标准化）
    4. 填充/截断到固定长度 (1,228,800)
    5. 重塑为 (1200, 1024)
    
    Returns:
        processed_ppg: (1200, 1024)
        processed_ecg: (1200, 1024)
        labels: (1200,)
        subject_id: 受试者ID
        n_epochs: 原始epoch数
        action: 'truncate' 或 'pad'
    """
    print(f"\n{'='*60}")
    print(f"Processing subject_{subject_num:02d}...")
    print(f"{'='*60}")
    
    # 加载原始数据
    ppg_df, ecg_df, ann_df, subject_id = load_subject_data(subject_num)
    
    # 获取原始epoch数
    n_epochs = len(ann_df)
    print(f"  Original epochs: {n_epochs}")
    
    # 提取并合并为连续信号
    ppg_continuous = extract_signals_from_df(ppg_df, 'ppg')
    ecg_continuous = extract_signals_from_df(ecg_df, 'ecg')
    
    print(f"  Original PPG length: {len(ppg_continuous)} samples @ {PPG_FS_ORIGINAL}Hz")
    print(f"  Original ECG length: {len(ecg_continuous)} samples @ {ECG_FS_ORIGINAL}Hz")
    print(f"  Duration: ~{len(ppg_continuous) / PPG_FS_ORIGINAL / 60:.1f} minutes")
    
    # 获取标签
    stages = ann_df['Sleep Stage'].values
    
    # 映射标签
    labels = []
    for stage in stages:
        if stage in STAGE_MAP:
            labels.append(STAGE_MAP[stage])
        else:
            labels.append(-1)  # 未知标签
    labels = np.array(labels)
    
    # 预处理 PPG（整个连续信号）
    print(f"  Preprocessing PPG...")
    processed_ppg = preprocess_ppg(ppg_continuous, PPG_FS_ORIGINAL, TARGET_FS)
    print(f"  After preprocessing: {len(processed_ppg)} samples @ {TARGET_FS}Hz")
    
    # 预处理 ECG（整个连续信号）
    print(f"  Preprocessing ECG...")
    processed_ecg = preprocess_ecg(ecg_continuous, ECG_FS_ORIGINAL, TARGET_FS)
    print(f"  After preprocessing: {len(processed_ecg)} samples @ {TARGET_FS}Hz")
    
    # 填充/截断到目标长度 (1,228,800)
    print(f"  Target length: {TARGET_LENGTH} samples (1200 windows × 1024 samples)")
    
    processed_ppg, ppg_action, ppg_diff = pad_or_truncate(processed_ppg, TARGET_LENGTH)
    processed_ecg, ecg_action, ecg_diff = pad_or_truncate(processed_ecg, TARGET_LENGTH)
    
    # 确定总体action（优先显示截断，因为数据更重要）
    if n_epochs >= WINDOWS_PER_SUBJECT:
        action = 'truncate'
        diff = n_epochs - WINDOWS_PER_SUBJECT
        print(f"  >>> Action: TRUNCATE (原始{n_epochs}个epoch > 目标{WINDOWS_PER_SUBJECT}个epoch)")
        print(f"      将截断最后 {diff} 个epoch的数据")
    else:
        action = 'pad'
        diff = WINDOWS_PER_SUBJECT - n_epochs
        print(f"  >>> Action: PAD (原始{n_epochs}个epoch < 目标{WINDOWS_PER_SUBJECT}个epoch)")
        print(f"      将填充 {diff} 个epoch的零值数据")
    
    # 处理标签
    labels, label_action, label_diff = pad_labels(labels, WINDOWS_PER_SUBJECT, n_epochs)
    
    # 重塑为窗口格式 (1200, 1024)
    processed_ppg = processed_ppg.reshape(WINDOWS_PER_SUBJECT, SAMPLES_PER_WINDOW)
    processed_ecg = processed_ecg.reshape(WINDOWS_PER_SUBJECT, SAMPLES_PER_WINDOW)
    
    # 统计信息
    print(f"\n  Final PPG shape: {processed_ppg.shape}")
    print(f"  Final ECG shape: {processed_ecg.shape}")
    print(f"  Final labels shape: {labels.shape}")
    
    # 统计有效标签分布
    valid_labels = labels[labels >= 0]
    if len(valid_labels) > 0:
        print(f"  Valid labels distribution:")
        stage_names = ['Wake', 'Light', 'Deep', 'REM']
        for i in range(4):
            count = np.sum(valid_labels == i)
            print(f"    {stage_names[i]} (class {i}): {count} epochs")
    
    # 无效标签数量
    invalid_count = np.sum(labels == -1)
    if invalid_count > 0:
        print(f"  Invalid/Padded labels: {invalid_count} epochs (marked as -1)")
    
    return processed_ppg, processed_ecg, labels, subject_id, n_epochs, action


def main():
    """主处理流程"""
    print("=" * 60)
    print("AAUWSS Data Extraction (Aligned with DREAMT/MESA format)")
    print("=" * 60)
    print(f"Target sampling rate: {TARGET_FS} Hz")
    print(f"Samples per window: {SAMPLES_PER_WINDOW}")
    print(f"Window duration: {WINDOW_DURATION} seconds")
    print(f"Windows per subject: {WINDOWS_PER_SUBJECT}")
    print(f"Total samples per subject: {TARGET_LENGTH}")
    print("=" * 60)
    
    all_ppg = []
    all_ecg = []
    all_labels = []
    subject_ids = []
    subject_indices = []
    current_idx = 0
    
    # 处理统计
    truncate_count = 0
    pad_count = 0
    subject_stats = []
    
    # 处理所有受试者
    subject_list = list(range(1, 14))  # subject_01 到 subject_13
    
    for subject_num in subject_list:
        try:
            ppg, ecg, labels, subject_id, n_epochs, action = process_subject(subject_num)
            
            all_ppg.append(ppg)
            all_ecg.append(ecg)
            all_labels.append(labels)
            
            # 记录受试者ID（每个窗口对应一个subject_id）
            subject_ids.extend([subject_id] * WINDOWS_PER_SUBJECT)
            
            # 记录受试者索引（与DREAMT格式一致）
            subject_indices.append({
                'subject_id': subject_id,
                'start_idx': current_idx,
                'end_idx': current_idx + WINDOWS_PER_SUBJECT,
                'n_windows': WINDOWS_PER_SUBJECT,  # 固定为1200
                'original_epochs': n_epochs,
                'action': action
            })
            current_idx += WINDOWS_PER_SUBJECT
            
            # 统计
            if action == 'truncate':
                truncate_count += 1
            else:
                pad_count += 1
            
            subject_stats.append({
                'subject_id': subject_id,
                'original_epochs': n_epochs,
                'action': action
            })
            
        except Exception as e:
            print(f"  Error processing subject_{subject_num:02d}: {e}")
            continue
    
    # 打印处理统计摘要
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total subjects processed: {len(subject_stats)}")
    print(f"Subjects truncated (epochs > 1200): {truncate_count}")
    print(f"Subjects padded (epochs < 1200): {pad_count}")
    print("\nPer-subject details:")
    print("-" * 60)
    print(f"{'Subject':<12} {'Original Epochs':<18} {'Action':<10} {'Target':<10}")
    print("-" * 60)
    for stat in subject_stats:
        print(f"{stat['subject_id']:<12} {stat['original_epochs']:<18} {stat['action']:<10} 1200")
    print("-" * 60)
    
    # 合并所有数据
    print("\n" + "=" * 60)
    print("Merging all subjects...")
    print("=" * 60)
    
    all_ppg = np.concatenate(all_ppg, axis=0)
    all_ecg = np.concatenate(all_ecg, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    subject_ids = np.array(subject_ids)
    
    print(f"Total PPG shape: {all_ppg.shape}")
    print(f"Total ECG shape: {all_ecg.shape}")
    print(f"Total labels shape: {all_labels.shape}")
    print(f"Expected shape: ({len(subject_stats) * WINDOWS_PER_SUBJECT}, {SAMPLES_PER_WINDOW})")
    
    # 统计标签分布
    valid_labels = all_labels[all_labels >= 0]
    print(f"\nValid labels distribution:")
    stage_names = ['Wake', 'Light', 'Deep', 'REM']
    for i in range(4):
        count = np.sum(valid_labels == i)
        percentage = count / len(valid_labels) * 100 if len(valid_labels) > 0 else 0
        print(f"  {stage_names[i]} (class {i}): {count} ({percentage:.2f}%)")
    
    invalid_count = np.sum(all_labels == -1)
    print(f"  Invalid/Padded labels: {invalid_count}")
    
    # 保存数据
    print("\n" + "=" * 60)
    print("Saving processed data...")
    print("=" * 60)
    
    # 保存 PPG + 标签（与DREAMT格式一致）
    ppg_output_path = os.path.join(OUTPUT_DIR, "aauwss_ppg_with_labels.h5")
    with h5py.File(ppg_output_path, 'w') as f:
        f.create_dataset('ppg', data=all_ppg, compression='gzip',
                         chunks=(100, SAMPLES_PER_WINDOW))
        f.create_dataset('labels', data=all_labels, compression='gzip')
        f.create_dataset('subject_ids', data=subject_ids.astype('S10'), compression='gzip')
        
        # 添加元数据（与DREAMT一致）
        f.attrs['sampling_rate'] = TARGET_FS
        f.attrs['window_duration'] = WINDOW_DURATION
        f.attrs['samples_per_window'] = SAMPLES_PER_WINDOW
        f.attrs['windows_per_subject'] = WINDOWS_PER_SUBJECT
        f.attrs['total_windows'] = len(all_ppg)
        f.attrs['total_subjects'] = len(subject_stats)
        f.attrs['data_source'] = 'AAUWSS'
        f.attrs['original_ppg_fs'] = PPG_FS_ORIGINAL
        f.attrs['original_ecg_fs'] = ECG_FS_ORIGINAL
        f.attrs['truncate_count'] = truncate_count
        f.attrs['pad_count'] = pad_count
    print(f"Saved: {ppg_output_path}")
    
    # 保存 ECG（与DREAMT格式一致）
    ecg_output_path = os.path.join(OUTPUT_DIR, "aauwss_ecg.h5")
    with h5py.File(ecg_output_path, 'w') as f:
        f.create_dataset('ecg', data=all_ecg, compression='gzip',
                         chunks=(100, SAMPLES_PER_WINDOW))
        f.create_dataset('subject_ids', data=subject_ids.astype('S10'), compression='gzip')
        f.create_dataset('labels', data=all_labels, compression='gzip')
    print(f"Saved: {ecg_output_path}")
    
    # 保存受试者索引（与DREAMT格式完全一致）
    index_output_path = os.path.join(OUTPUT_DIR, "aauwss_subject_index.h5")
    with h5py.File(index_output_path, 'w') as f:
        subject_group = f.create_group('subjects')
        for idx_info in subject_indices:
            grp = subject_group.create_group(idx_info['subject_id'])
            # window_indices: 每个受试者的窗口索引范围
            grp.create_dataset('window_indices', 
                               data=np.arange(idx_info['start_idx'], idx_info['end_idx']))
            # n_windows: 固定为1200（与DREAMT一致）
            grp.attrs['n_windows'] = idx_info['n_windows']
            grp.attrs['original_epochs'] = idx_info['original_epochs']
            grp.attrs['action'] = idx_info['action']
        
        f.attrs['total_subjects'] = len(subject_indices)
        f.attrs['total_windows'] = len(all_ppg)
        f.attrs['windows_per_subject'] = WINDOWS_PER_SUBJECT
    print(f"Saved: {index_output_path}")
    
    print("\n" + "=" * 60)
    print("AAUWSS Data Extraction Complete!")
    print("=" * 60)
    print(f"Output format: ({len(all_ppg)}, {SAMPLES_PER_WINDOW})")
    print(f"Per subject: ({WINDOWS_PER_SUBJECT}, {SAMPLES_PER_WINDOW})")
    print(f"Total subjects: {len(subject_stats)}")
    print(f"Format: Fully aligned with DREAMT/MESA format")
    print("=" * 60)
    
    return all_ppg, all_ecg, all_labels, subject_indices, subject_stats


if __name__ == "__main__":
    main()