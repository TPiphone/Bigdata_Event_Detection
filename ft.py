import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq

SAMPLE_RATE = 44100  # Hertz
DURATION = 5  # Seconds

def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = x * freq
    # 2pi because np.sin takes radians
    y = np.sin((2 * np.pi) * frequencies)
    return x, y

_, nice_tone = generate_sine_wave(400, SAMPLE_RATE, DURATION)
_, noise_tone = generate_sine_wave(4000, SAMPLE_RATE, DURATION)
noise_tone = noise_tone * 0.3

mixed_tone = nice_tone + noise_tone

normalized_tone = np.int16((mixed_tone / mixed_tone.max()) * 32767)

# plt.plot(normalized_tone[:1000])
# plt.show()


# Number of samples in normalized_tone
N = SAMPLE_RATE * DURATION

print(normalized_tone)
print(normalized_tone.shape)
yf = fft(normalized_tone)
xf = fftfreq(N, 1 / SAMPLE_RATE)

plt.plot(xf, np.abs(yf))
# plt.show()

# # Find the numbers that occur more than once
unique_elements, counts = np.unique(normalized_tone, return_counts=True)
duplicates = unique_elements[counts > 1]
print("Numbers that occur more than once:", len(duplicates))

