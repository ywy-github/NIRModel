#导入库
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

#导入数据
(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

#图像处理，增加维度
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
# 标准化图片到区间 [0., 1.] 内
train_images /= 255.
test_images /= 255.
# 二值化
train_images[train_images >= .5] = 1.
train_images[train_images < .5] = 0.
test_images[test_images >= .5] = 1.
test_images[test_images < .5] = 0.

#数据集
TRAIN_BUF = 60000
BATCH_SIZE = 100
TEST_BUF = 10000

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)


# 创建CVAE模型

class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(latent_dim + latent_dim)
            ]
        )

        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding='same',
                                                activation='relu'),
                tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2), padding='same',
                                                activation='relu'),
                tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), padding='same'),
            ]
        )

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)

        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

#优化器
optimizer = tf.keras.optimizers.Adam(1e-4)

#正态分布的概率密度函数的对数
def log_normal_pdf(sample,mean,logvar,raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(-.5 * ((sample-mean) ** 2. * tf.exp(-logvar)+logvar+log2pi),axis=raxis)


# 计算损失
def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    cross_ent = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x), axis=[1, 2, 3])
    KLD = -0.5 * tf.reduce_sum(1 + logvar - tf.pow(mean, 2) - tf.exp(logvar), axis=-1)

    return tf.reduce_mean(cross_ent + KLD)
#定义梯度下降
def compute_apply_gradients(model, x, optimizer):
  with tf.GradientTape() as tape:
    loss = compute_loss(model,x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#定义显示图像函数
def generate_and_save_images(model, epoch, test_input):
  predictions = model.sample(test_input)
  fig = plt.figure(figsize=(4,4))
  for i in range(predictions.shape[0]):
    plt.subplot(4,4,i+1)
    plt.imshow(predictions[i,:,:,0],cmap='gray')
    plt.axis('off')
  plt.savefig('img_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

epochs = 100
latent_dim = 50
num_examples_to_generate = 16
# 保持随机向量恒定以进行生成（预测），以便更易于看到改进。
random_vector_for_generation = tf.random.normal(
    shape = [num_examples_to_generate, latent_dim]
)
model = CVAE(latent_dim)

generate_and_save_images(model,0,random_vector_for_generation)

for epoch in range(1,epochs+1):
  for train_x in train_dataset:
    compute_apply_gradients(model,train_x,optimizer)

    if epoch % 1 == 0:
      loss = tf.keras.metrics.Mean()
      for test_x in test_dataset:
        loss(compute_loss(model,test_x))
      elbo = loss.result()
      display.clear_output(wait=False)
      print('Epoch:{},Test set ELBO:{},'.format(epoch,elbo))
  generate_and_save_images(model,epoch,random_vector_for_generation)


