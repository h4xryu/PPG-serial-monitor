import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import lfilter


def lpc(signal, order):
    """
    주어진 음성 신호에서 LPC 계수를 계산합니다.

    signal: 입력 음성 신호 (1D numpy array)
    order: LPC 모델의 차수 (계수 개수)
    """
    r = np.correlate(signal, signal, mode="full")
    r = r[len(r) // 2 :]  # 자기상관 함수

    a, e = levinson_durbin(r, order)
    return a


def levinson_durbin(r, order):
    """
    Levinson-Durbin 알고리즘을 사용하여 AR 계수(LPC 계수)를 계산합니다.
    """
    a = np.zeros(order + 1)  # AR 계수
    e = np.zeros(order + 1)  # 예측 에너지

    # 첫 번째 단계
    a[1] = -r[1] / r[0]
    e[1] = r[0] + a[1] * r[1]

    for p in range(2, order + 1):
        acc = sum([a[j] * r[p - j] for j in range(1, p)])
        k = -(r[p] + acc) / e[p - 1]

        new_a = a.copy()
        for j in range(1, p):
            new_a[j] = a[j] + k * a[p - j]
        new_a[p] = k

        a = new_a
        e[p] = e[p - 1] * (1 - k**2)

    return a[1:], e[order]


def synthesize(signal, lpc_coeffs):
    """
    LPC 계수를 사용해 음성 신호를 재구성합니다.

    signal: 입력 음성 신호 (1D numpy array)
    lpc_coeffs: LPC 계수
    """
    # 신호 예측 (filtering)
    return lfilter([1], np.concatenate([[1], -lpc_coeffs]), signal)


# 음성 신호 로드
signal, sr = librosa.load(librosa.example("libri1"), sr=16000)

# 30ms 프레임으로 분할
frame_size = int(0.03 * sr)  # 30ms 프레임
frame = signal[1000 : 1000 + frame_size]  # 30ms 동안의 신호 추출

# LPC 계수 계산
order = 12  # LPC 차수
lpc_coeffs = lpc(frame, order)

# 원 신호 재구성
synthesized_signal = synthesize(frame, lpc_coeffs)

# 결과 시각화
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(frame, label="Original Signal")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(synthesized_signal, label="Synthesized Signal")
plt.legend()
plt.show()
