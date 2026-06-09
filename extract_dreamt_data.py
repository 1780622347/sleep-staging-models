import os
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt
import h5py
from tqdm import tqdm
import warnings
from collections import Counter

warnings.filterwarnings('ignore')


class DREAMTDataExtractor:
    def __init__(self, dreamt_path, output_path, use_100hz=True):
        """
        DREAMT数据提取器
        
        Args:
            dreamt_path: DREAMT数据集根目录路径
            output_path: 输出文件路径
            use_100hz: 是否使用100Hz数据（包含ECG），False则使用64Hz数据
        """
        self.dreamt_path = dreamt_path
        self.output_path = output_path
        self.use_100hz = use_100hz
        
        # 沿用MESA的处理参数
        self.target_fs = 34.133333333  # 目标采样率
        self.window_duration = 30  # 窗口时长（秒）
        self.samples_per_window = 1024  # 每个窗口的采样点数
        self.target_hours = 10  # 目标时长（小时）
        self.target_windows = 1200  # 目标窗口数
        self.target_length = self.target_windows * self.samples_per_window  # 1,228,800 samples
        
        # DREAMT数据采样率
        self.ppg_fs = 64  # BVP采样率
        self.ecg_fs = 100 if use_100hz else None  # ECG采样率（仅100Hz数据有）
        
        # 标签映射（DREAMT标签 -> 4分类）
        self.stage_map = {
            'W': 0,    # Wake
            'N1': 1,   # Light sleep
            'N2': 1,   # Light sleep
            'N3': 2,   # Deep sleep
            'R': 3,    # REM
        }
        
        os.makedirs(output_path, exist_ok=True)
        
        # 确定数据目录
        if use_100hz:
            self.data_dir = os.path.join(dreamt_path, 'data_100Hz')
        else:
            self.data_dir = os.path.join(dreamt_path, 'data_64Hz')
    
    def extract_signals_from_csv(self, csv_file):
        """
        从CSV文件中提取PPG(BVP)和ECG信号
        
        Returns:
            ppg_signal, ecg_signal, sleep_stages
        """
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            
            # 提取BVP作为PPG信号
            if 'BVP' not in df.columns:
                print(f"No BVP column found in {os.path.basename(csv_file)}")
                return None, None, None
            
            ppg_signal = df['BVP'].values
            
            # 提取ECG信号（仅100Hz数据有）
            ecg_signal = None
            if self.use_100hz and 'ECG' in df.columns:
                ecg_signal = df['ECG'].values
            
            # 提取睡眠分期标签
            if 'Sleep_Stage' not in df.columns:
                print(f"No Sleep_Stage column found in {os.path.basename(csv_file)}")
                return None, None, None
            
            sleep_stages = df['Sleep_Stage'].values
            
            return ppg_signal, ecg_signal, sleep_stages
            
        except Exception as e:
            print(f"Error reading {os.path.basename(csv_file)}: {e}")
            return None, None, None
    
    def preprocess_ppg(self, ppg_signal, original_fs=64):
        """
        PPG预处理 - 沿用MESA的处理流程
        
        步骤:
        1. 8Hz低通滤波（抗混叠）
        2. 重采样到34.13Hz
        3. 裁剪到mean±3std
        4. Z-score标准化
        """
        # 处理NaN值
        ppg_signal = pd.Series(ppg_signal).fillna(method='ffill').fillna(method='bfill').values
        
        # 8Hz低通滤波（Chebyshev II型）
        nyq = 0.5 * original_fs
        cutoff = 8 / nyq
        sos = signal.cheby2(N=8, rs=40, Wn=cutoff, btype='lowpass', output='sos')
        filtered_ppg = signal.sosfiltfilt(sos, ppg_signal)
        
        # 重采样到目标频率34.13Hz
        duration = len(filtered_ppg) / original_fs
        n_samples = int(duration * self.target_fs)
        
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
    
    def preprocess_ecg(self, ecg_signal, original_fs=100):
        """
        ECG预处理 - 沿用MESA的处理流程
        
        步骤:
        1. 0.5-40Hz带通滤波
        2. 重采样到34.13Hz
        3. 裁剪到mean±3std
        4. Z-score标准化
        """
        # 处理NaN值
        ecg_signal = pd.Series(ecg_signal).fillna(method='ffill').fillna(method='bfill').values
        
        # 0.5-40Hz带通滤波
        nyq = 0.5 * original_fs
        low = 0.5 / nyq
        high = 40.0 / nyq
        
        if high >= 1:
            high = 0.99
        
        b, a = butter(4, [low, high], btype='band')
        filtered_ecg = filtfilt(b, a, ecg_signal)
        
        # 重采样到目标频率34.13Hz
        duration = len(filtered_ecg) / original_fs
        n_samples = int(duration * self.target_fs)
        
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
    
    def pad_or_truncate_signal(self, signal, target_length):
        """
        填充或截断信号到目标长度
        """
        current_length = len(signal)
        
        if current_length >= target_length:
            return signal[:target_length]
        else:
            padding_length = target_length - current_length
            padding = np.zeros(padding_length)
            return np.concatenate([signal, padding])
    
    def process_sleep_stages(self, sleep_stages, signal_length, original_fs=64):
        """
        处理睡眠分期标签
        
        将每个采样点的标签转换为30秒窗口标签
        """
        # 映射标签
        mapped_stages = np.full(len(sleep_stages), -1, dtype=int)
        
        for i, stage in enumerate(sleep_stages):
            if pd.isna(stage) or stage == 'Missing' or stage == 'P':
                continue
            if stage in self.stage_map:
                mapped_stages[i] = self.stage_map[stage]
        
        # 计算窗口数
        samples_per_window_original = int(original_fs * self.window_duration)
        num_windows = int(np.ceil(len(sleep_stages) / samples_per_window_original))
        
        # 为每个窗口分配标签（取众数）
        window_labels = np.full(num_windows, -1, dtype=int)
        
        for w in range(num_windows):
            start_idx = w * samples_per_window_original
            end_idx = min((w + 1) * samples_per_window_original, len(mapped_stages))
            
            window_vals = mapped_stages[start_idx:end_idx]
            valid_vals = window_vals[window_vals != -1]
            
            if len(valid_vals) > 0:
                # 取众数作为窗口标签
                window_labels[w] = Counter(valid_vals).most_common(1)[0][0]
        
        return window_labels
    
    def pad_or_truncate_labels(self, labels, target_length):
        """
        填充或截断标签到目标长度
        """
        current_length = len(labels)
        
        if current_length >= target_length:
            return labels[:target_length]
        else:
            padding_length = target_length - current_length
            padding = np.full(padding_length, -1, dtype=int)
            return np.concatenate([labels, padding])
    
    def process_subject(self, csv_file):
        """
        处理单个受试者数据
        
        Returns:
            ppg_windows, ecg_windows, labels_final, has_real_ecg
        """
        # 提取信号
        ppg_signal, ecg_signal, sleep_stages = self.extract_signals_from_csv(csv_file)
        
        if ppg_signal is None:
            return None
        
        # 预处理PPG
        ppg_processed = self.preprocess_ppg(ppg_signal, self.ppg_fs)
        
        # 预处理ECG
        if ecg_signal is not None:
            ecg_processed = self.preprocess_ecg(ecg_signal, self.ecg_fs)
            has_real_ecg = True
        else:
            ecg_processed = np.zeros_like(ppg_processed)
            has_real_ecg = False
        
        # 填充/截断到目标长度
        ppg_final = self.pad_or_truncate_signal(ppg_processed, self.target_length)
        ecg_final = self.pad_or_truncate_signal(ecg_processed, self.target_length)
        
        # 处理睡眠分期标签
        labels = self.process_sleep_stages(sleep_stages, len(ppg_signal), self.ppg_fs)
        labels_final = self.pad_or_truncate_labels(labels, self.target_windows)
        
        # 重塑为窗口形式
        try:
            ppg_windows = ppg_final.reshape(self.target_windows, self.samples_per_window)
            ecg_windows = ecg_final.reshape(self.target_windows, self.samples_per_window)
        except ValueError as e:
            print(f"Reshape error: {e}")
            print(f"PPG shape: {ppg_final.shape}, expected: ({self.target_windows}, {self.samples_per_window})")
            return None
        
        return ppg_windows, ecg_windows, labels_final, has_real_ecg
    
    def process_all_subjects(self, subject_list=None):
        """
        处理所有受试者
        """
        # 获取所有CSV文件
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        if subject_list:
            csv_files = [f for f in csv_files if any(subj in f for subj in subject_list)]
        
        # 排序文件
        csv_files.sort()
        
        all_ppg_windows = []
        all_ecg_windows = []
        all_labels = []
        subject_ids = []
        has_real_ecg_list = []
        
        subjects_with_ecg = 0
        subjects_without_ecg = 0
        failed_subjects = 0
        
        for csv_filename in tqdm(csv_files, desc="Processing subjects"):
            csv_path = os.path.join(self.data_dir, csv_filename)
            
            # 提取受试者ID
            # 文件名格式: S002_PSG_df.csv 或 S002_whole_df.csv
            subject_id = csv_filename.split('_')[0]  # S002
            
            try:
                result = self.process_subject(csv_path)
                
                if result is not None:
                    ppg_windows, ecg_windows, labels, has_ecg = result
                    
                    if has_ecg:
                        subjects_with_ecg += 1
                    else:
                        subjects_without_ecg += 1
                    
                    all_ppg_windows.append(ppg_windows)
                    all_ecg_windows.append(ecg_windows)
                    all_labels.append(labels)
                    subject_ids.extend([subject_id] * len(ppg_windows))
                    has_real_ecg_list.extend([has_ecg] * len(ppg_windows))
                else:
                    failed_subjects += 1
            except Exception as e:
                print(f"Error processing {csv_filename}: {e}")
                failed_subjects += 1
                continue
        
        print(f"\nProcessing summary:")
        print(f"  Subjects with ECG: {subjects_with_ecg}")
        print(f"  Subjects without ECG: {subjects_without_ecg}")
        print(f"  Failed subjects: {failed_subjects}")
        print(f"  Total subjects processed: {subjects_with_ecg + subjects_without_ecg}")
        
        # 合并所有数据
        if all_ppg_windows:
            all_ppg_windows = np.vstack(all_ppg_windows)
            all_ecg_windows = np.vstack(all_ecg_windows)
            all_labels = np.concatenate(all_labels)
            subject_ids = np.array(subject_ids)
            has_real_ecg = np.array(has_real_ecg_list)
            
            # 保存数据
            self.save_data_separate(all_ppg_windows, all_ecg_windows, all_labels, subject_ids, has_real_ecg)
            
            return len(all_ppg_windows)
        
        return 0
    
    def save_data_separate(self, ppg_windows, ecg_windows, labels, subject_ids, has_real_ecg):
        """
        保存数据到H5文件
        """
        # 保存PPG数据
        ppg_file = os.path.join(self.output_path, 'dreamt_ppg_with_labels.h5')
        print(f"\nSaving PPG data to {ppg_file}...")
        with h5py.File(ppg_file, 'w') as f:
            f.create_dataset('ppg', data=ppg_windows, compression='gzip',
                             chunks=(100, self.samples_per_window))
            f.create_dataset('labels', data=labels, compression='gzip')
            f.create_dataset('subject_ids', data=subject_ids.astype('S10'), compression='gzip')
            
            # 添加元数据
            f.attrs['sampling_rate'] = self.target_fs
            f.attrs['window_duration'] = self.window_duration
            f.attrs['samples_per_window'] = self.samples_per_window
            f.attrs['total_windows'] = len(ppg_windows)
            f.attrs['total_subjects'] = len(np.unique(subject_ids))
            f.attrs['data_source'] = 'DREAMT'
            f.attrs['original_ppg_fs'] = self.ppg_fs
            f.attrs['original_ecg_fs'] = self.ecg_fs if self.ecg_fs else 'N/A'
        
        # 保存ECG数据
        ecg_file = os.path.join(self.output_path, 'dreamt_ecg.h5')
        print(f"Saving ECG data to {ecg_file}...")
        with h5py.File(ecg_file, 'w') as f:
            f.create_dataset('ecg', data=ecg_windows, compression='gzip',
                             chunks=(100, self.samples_per_window))
            f.create_dataset('has_real_ecg', data=has_real_ecg, compression='gzip')
            f.create_dataset('subject_ids', data=subject_ids.astype('S10'), compression='gzip')
            f.create_dataset('labels', data=labels, compression='gzip')
            
            # 记录有真实ECG的索引
            real_ecg_indices = np.where(has_real_ecg)[0]
            f.create_dataset('real_ecg_indices', data=real_ecg_indices, compression='gzip')
            f.attrs['windows_with_real_ecg'] = int(np.sum(has_real_ecg))
            f.attrs['windows_without_real_ecg'] = int(np.sum(~has_real_ecg))
        
        # 保存受试者索引
        index_file = os.path.join(self.output_path, 'dreamt_subject_index.h5')
        print(f"Creating index file {index_file}...")
        with h5py.File(index_file, 'w') as f:
            unique_subjects = np.unique(subject_ids)
            subject_group = f.create_group('subjects')
            
            for subj in unique_subjects:
                subj_str = subj.decode() if isinstance(subj, bytes) else str(subj)
                mask = subject_ids == subj
                indices = np.where(mask)[0]
                
                subj_group = subject_group.create_group(subj_str)
                subj_group.create_dataset('window_indices', data=indices)
                subj_group.attrs['n_windows'] = len(indices)
                subj_group.attrs['has_ecg'] = bool(has_real_ecg[indices[0]])
            
            f.attrs['total_subjects'] = len(unique_subjects)
            f.attrs['total_windows'] = len(ppg_windows)
        
        # 保存统计信息
        self.save_statistics(ppg_windows, ecg_windows, labels, subject_ids, has_real_ecg)
        
        print("\nData saved successfully in separate files!")
        print(f"  PPG data: {ppg_file}")
        print(f"  ECG data: {ecg_file}")
        print(f"  Subject index: {index_file}")
    
    def save_statistics(self, ppg_windows, ecg_windows, labels, subject_ids, has_real_ecg):
        """
        保存统计信息
        """
        valid_labels = labels[labels != -1]
        
        stats = {
            'total_windows': len(ppg_windows),
            'total_subjects': len(np.unique(subject_ids)),
            'windows_with_real_ecg': int(np.sum(has_real_ecg)),
            'windows_without_real_ecg': int(np.sum(~has_real_ecg)),
            'ppg_shape': ppg_windows.shape,
            'ecg_shape': ecg_windows.shape,
            'valid_labels': len(valid_labels),
            'label_distribution': dict(zip(*np.unique(valid_labels, return_counts=True))) if len(valid_labels) > 0 else {},
            'sampling_rate': self.target_fs,
            'window_duration': self.window_duration,
            'samples_per_window': self.samples_per_window,
            'data_source': 'DREAMT',
            'original_ppg_fs': self.ppg_fs,
            'original_ecg_fs': self.ecg_fs if self.ecg_fs else 'N/A',
            'file_structure': {
                'dreamt_ppg_with_labels.h5': ['ppg', 'labels', 'subject_ids'],
                'dreamt_ecg.h5': ['ecg', 'has_real_ecg', 'subject_ids', 'labels', 'real_ecg_indices'],
                'dreamt_subject_index.h5': ['subjects/{subject_id}/window_indices']
            }
        }
        
        # 保存numpy格式统计
        stats_file = os.path.join(self.output_path, 'data_stats.npy')
        np.save(stats_file, stats)
        
        # 保存文本格式统计
        stats_txt = os.path.join(self.output_path, 'data_stats.txt')
        with open(stats_txt, 'w', encoding='utf-8') as f:
            f.write("DREAMT Sleep Data Processing Statistics\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total windows: {stats['total_windows']}\n")
            f.write(f"Total subjects: {stats['total_subjects']}\n")
            f.write(f"Windows with real ECG: {stats['windows_with_real_ecg']}\n")
            f.write(f"Windows without real ECG: {stats['windows_without_real_ecg']}\n")
            f.write(f"Valid labels: {stats['valid_labels']}\n")
            f.write(f"\nOriginal sampling rates:\n")
            f.write(f"  PPG: {self.ppg_fs} Hz\n")
            f.write(f"  ECG: {self.ecg_fs if self.ecg_fs else 'N/A'} Hz\n")
            f.write(f"\nTarget sampling rate: {stats['sampling_rate']} Hz\n")
            f.write(f"Window duration: {stats['window_duration']} seconds\n")
            f.write(f"Samples per window: {stats['samples_per_window']}\n")
            f.write(f"\nLabel distribution:\n")
            stage_names = {0: 'Wake', 1: 'Light', 2: 'Deep', 3: 'REM'}
            for label, count in stats['label_distribution'].items():
                f.write(f"  {stage_names.get(label, f'Stage{label}')}: {count}\n")
        
        print(f"\nStatistics saved to:")
        print(f"  {stats_file}")
        print(f"  {stats_txt}")


def main():
    # DREAMT数据集路径
    DREAMT_PATH = r"D:\工作记录\算法测试20260413\Data_Sets\sleep\dreamt-dataset"
    
    # 输出路径
    OUTPUT_PATH = r"D:\工作记录\算法测试20260413\sleep_staging\sleep-staging-models\dreamt_processed"
    
    # 创建提取器（使用100Hz数据以获取ECG）
    extractor = DREAMTDataExtractor(DREAMT_PATH, OUTPUT_PATH, use_100hz=True)
    
    # 处理所有受试者
    n_windows = extractor.process_all_subjects()
    
    print(f"\nProcessing completed! Total windows: {n_windows}")
    
    # 如果需要处理特定受试者，可以使用：
    # n_windows = extractor.process_all_subjects(subject_list=['S002', 'S003', 'S004'])


if __name__ == "__main__":
    main()