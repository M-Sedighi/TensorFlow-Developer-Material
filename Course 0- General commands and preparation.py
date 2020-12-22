import numpy as np
import tensorflow as tf
import sys

# ==================get the array shape===================================

arr = np.array([[1, 2, 3], [1, 2, 1]])
print(arr.shape)  # to get the dimensionality of the data

# ==================check GPU usage in TensorFlow=========================

print(tf.__version__)

# test for GPU and Cuda
tf.compat.v1.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)

# this is to confirm that the GPU is being used
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

# ==================Clear data and models in Keras=========================

tf.keras.backend.clear_session()  # to clear keras data


# ==================Bar progress function for Wget=========================

def bar_progress(current, total, width):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    # Don't use print() as it will print in new line every time.
    if width == width:
        sys.stdout.write("\r" + progress_message)
        sys.stdout.flush()


# ==================function for plotting time series======================
def plot_series(t, seris, frmt="-", start=0, end=None):
    plt.plot(t[start:end], seris[start:end], frmt)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


# ==================windows data set and forcasting helper functions (3d)==

def windowed_dataset(seris, wndw_size, btch_size, shuffle_buffer):
    seris = tf.expand_dims(seris, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(seris)
    ds = ds.window(wndw_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(wndw_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(btch_size).prefetch(1)


def model_forecast(model, seris, wndw_size):
    ds = tf.data.Dataset.from_tensor_slices(seris)
    ds = ds.window(wndw_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(wndw_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast

