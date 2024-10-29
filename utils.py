import numpy as np
from scipy.signal import cheby2, filtfilt, resample
from sklearn.preprocessing import StandardScaler


# 1. 저역 통과 필터 정의 (Chebyshev Type II)
def lowpass_filter(ppg_signal, sampling_rate, cutoff_freq=8, stopband_atten=40):
    """
    Chebyshev Type II 저역 통과 필터를 사용하여 PPG 신호를 필터링합니다.

    Args:
        ppg_signal (array): 원본 PPG 신호
        sampling_rate (int): 원본 신호의 샘플링 속도 (Hz)
        cutoff_freq (int): 필터의 컷오프 주파수 (Hz)
        stopband_atten (int): 스톱밴드 감쇠량 (dB)

    Returns:
        filtered_signal (array): 필터링된 PPG 신호
    """
    nyquist_rate = sampling_rate / 2
    normalized_cutoff = cutoff_freq / nyquist_rate
    b, a = cheby2(8, stopband_atten, normalized_cutoff, btype="low", analog=False)

    padlen = 3 * max(len(b), len(a))  # padlen을 계산
    if len(ppg_signal) <= padlen:
        raise ValueError(
            f"입력 신호의 길이가 {padlen}보다 짧습니다. 필터링을 적용할 수 없습니다."
        )

    # 필터 적용 (제로 위상 필터링)
    filtered_signal = filtfilt(b, a, ppg_signal, axis=0)
    return filtered_signal


# 2. 다운샘플링 (343Hz로 다운샘플링)
def downsample(ppg_signal, original_rate, target_rate=343):
    """
    주어진 PPG 신호를 목표 샘플링 속도로 다운샘플링합니다.

    Args:
        ppg_signal (array): 원본 PPG 신호
        original_rate (int): 원본 신호의 샘플링 속도 (Hz)
        target_rate (int): 다운샘플링할 목표 샘플링 속도 (Hz)

    Returns:
        downsampled_signal (array): 다운샘플링된 PPG 신호
    """
    num_samples = int(len(ppg_signal) * target_rate / original_rate)
    downsampled_signal = resample(ppg_signal, num_samples)
    return downsampled_signal


# 3. 클리핑 (표준편차 3배 이내 값으로 클리핑)
def clip_signal(ppg_signal, std_multiplier=3):
    """
    PPG 신호의 값을 표준편차 3배 이내로 클리핑합니다.

    Args:
        ppg_signal (array): 원본 PPG 신호
        std_multiplier (int): 표준편차 범위를 결정하는 값

    Returns:
        clipped_signal (array): 클리핑된 PPG 신호
    """
    mean_val = np.mean(ppg_signal)
    std_val = np.std(ppg_signal)
    clipped_signal = np.clip(
        ppg_signal,
        mean_val - std_multiplier * std_val,
        mean_val + std_multiplier * std_val,
    )
    return clipped_signal


# 4. 표준화 (평균을 빼고 표준편차로 나누기)
def standardize_signal(ppg_signal):
    """
    PPG 신호를 표준화합니다 (평균을 빼고 표준편차로 나누기).

    Args:
        ppg_signal (array): 원본 PPG 신호

    Returns:
        standardized_signal (array): 표준화된 PPG 신호
    """
    scaler = StandardScaler()
    standardized_signal = scaler.fit_transform(ppg_signal.reshape(-1, 1)).flatten()
    return standardized_signal


# 전체 전처리 함수
def preprocess_ppg(ppg_signal, original_rate):
    """
    PPG 신호에 대해 필터링, 다운샘플링, 클리핑, 표준화를 적용하는 전처리 함수.

    Args:
        ppg_signal (array): 원본 PPG 신호
        original_rate (int): 원본 신호의 샘플링 속도 (Hz)

    Returns:
        preprocessed_signal (array): 전처리된 PPG 신호
    """
    # 1. 저역 통과 필터 적용
    filtered_signal = lowpass_filter(ppg_signal, original_rate)

    # 2. 다운샘플링
    downsampled_signal = downsample(filtered_signal, original_rate)

    # 3. 클리핑
    clipped_signal = clip_signal(downsampled_signal)

    # 4. 표준화
    standardized_signal = standardize_signal(clipped_signal)

    return standardized_signal
