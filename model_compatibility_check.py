import tensorflow as tf
import tensorflow_hub as hub
"""
inp = tf.keras.layers.Input(shape=[1024, 1024, 3], name='input_image')
tar = tf.keras.layers.Input(shape=[1024, 1024, 3], name='target_image')

x = tf.keras.layers.concatenate([inp, tar], axis = 1)
discriminator_loaded = tf.keras.applications.DenseNet121(
                    include_top=False,
                    weights="imagenet",
                    input_tensor=None,
                    input_shape=(2048,1024,3),
                    pooling=None,
                    classes=1000,
                    )
out = discriminator_loaded(x)
discriminator = tf.keras.Model(inputs = [inp, tar], outputs = out)
discriminator.summary()
"""
module = hub.Module(r'C:\Users\Athrva Pandhare\Desktop\New folder (4)\sketch_to_art\gen_model')
#('https://tfhub.dev/deepmind/biggan-512/2')
module.summary()

# Sample random noise (z) and ImageNet label (y) inputs.
batch_size = 8
truncation = 0.5  # scalar truncation value in [0.02, 1.0]
z = truncation * tf.random.truncated_normal([batch_size, 128])  # noise sample
y_index = tf.random.uniform([batch_size], maxval=1000, dtype=tf.int32)
y = tf.one_hot(y_index, 1000)  # one-hot ImageNet label

# Call BigGAN on a dict of the inputs to generate a batch of images with shape
# [8, 512, 512, 3] and range [-1, 1].
samples = module(dict(y=y, z=z, truncation=truncation))
