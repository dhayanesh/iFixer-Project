from baseModel import DCGAN
from sklearn.utils import shuffle
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import torch
import numpy as np
import tqdm
import keras.backend as K


def perceptual_loss(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(64, 64, 3))
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


blurImages = (torch.load('path to train blurImages').numpy() - 127.5) / 127.5
sharpImages = (torch.load('path to train sharpImages').numpy() - 127.5) / 127.5

bt = (torch.load('path to test blurImages').numpy() - 127.5) / 127.5
st = torch.load('path to test sharpImages').numpy()

epoch_num = 5
batch_size = 256

shape = (64, 64, 3)

y_train, x_train = shuffle(sharpImages), shuffle(blurImages)

gen = DCGAN.build_generator(shape, 16, 3)
dis = DCGAN.build_discriminator(shape, 16)
inputs = Input(shape=shape)
gen_image = gen(inputs)
outputs = dis(gen_image)
dis_on_gen = Model(inputs=inputs, outputs=[gen_image, outputs])

dis_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
dis_on_gen_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

dis.trainable = True
dis.compile(optimizer=dis_opt, loss=wasserstein_loss)
dis.trainable = False
loss = [perceptual_loss, wasserstein_loss]
loss_weights = [100, 1]
dis_on_gen.compile(optimizer=dis_on_gen_opt, loss=loss, loss_weights=loss_weights)
dis.trainable = True

output_true_batch, output_false_batch = np.ones((batch_size, 1)), -np.ones((batch_size, 1))

for epoch in tqdm.tqdm(range(epoch_num)):

    print("[INFO] starting epoch {} of {}...".format(epoch + 1, epoch_num))

    dis_losses = []
    dis_on_gen_losses = []
    batchesPerEpoch = int(blurImages.shape[0] / batch_size)
    x_train = shuffle(x_train)
    y_train = shuffle(y_train)

    for i in range(batchesPerEpoch):

        image_blur_batch = x_train[i * batch_size:(i + 1) * batch_size]
        image_sharp_batch = y_train[i * batch_size:(i + 1) * batch_size]

        generated_images = gen.predict(x=image_blur_batch, batch_size=batch_size, verbose=0)

        for _ in range(5):
            dis_loss_real = dis.train_on_batch(image_sharp_batch, output_true_batch)
            dis_loss_fake = dis.train_on_batch(generated_images, output_false_batch)
            dis_loss = 0.5 * np.add(dis_loss_fake, dis_loss_real)
            dis_losses.append(dis_loss)

        dis.trainable = False

        dis_on_gen_loss = dis_on_gen.train_on_batch(image_blur_batch, [image_sharp_batch, output_true_batch])
        dis_on_gen_losses.append(dis_on_gen_loss)

        dis.trainable = True

    print(np.mean(dis_losses), np.mean(dis_on_gen_losses))
    with open('log.txt', 'a+') as f:
        f.write(
            'Epoch {} - Discriminator Loss {} - GaN Loss {}\n'.format(epoch, np.mean(dis_losses),
                                                                      np.mean(dis_on_gen_losses)))

pred = gen.predict(bt, verbose=0)
pred = (pred * 127.5) + 127.5

ss = []
pp = []
for i in range(len(pred)):
    ss.append(ssim(pred[i], st[i], multichannel=True))
    pp.append(psnr(pred[i], st[i], data_range=255))

ss = np.array(ss)
print(np.mean(ss))

pp = np.array(pp)
print(np.mean(pp))
