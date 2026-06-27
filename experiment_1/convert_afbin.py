# convert_afbin.py
import numpy as np

# Load raw after-FFT data
raw = np.fromfile('data/af.bin', dtype=np.complex64)
print(f"Loaded {len(raw)} samples")

# Slice into 288-length traces (same as training)
slice_len = 288
stride = 288  # non-overlapping, same as training
num_slices = (len(raw) - slice_len) // stride + 1

X = np.zeros((num_slices, slice_len), dtype=np.complex64)
for i in range(num_slices):
    X[i] = raw[i * stride : i * stride + slice_len]

# RMS normalize each trace
rms = np.sqrt(np.mean(np.abs(X)**2, axis=1, keepdims=True))
rms[rms == 0] = 1.0  # avoid divide by zero
X = X / rms

print(f"Created {X.shape[0]} traces of length {X.shape[1]}")
print(f"dtype: {X.dtype}")

np.savez('data/af_processed.npz', X=X)
print("Saved to data/af_processed.npz")