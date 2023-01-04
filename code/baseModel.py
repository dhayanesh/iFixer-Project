from tensorflow.keras.layers import Activation, Conv2D, BatchNormalization, Input, Layer, InputSpec, Add, Dropout, \
    Lambda, UpSampling2D, Flatten, Dense, LeakyReLU
from tensorflow.keras.models import Model
import tensorflow as tf
import keras.backend as K


class ReflectionPadding2D(Layer):

    def __init__(self, padding=(1, 1), data_format=None, **kwargs):

        if isinstance(padding, int):
            self.padding = tuple((padding, padding))
        else:
            self.padding = tuple(padding)
        if data_format is None:
            value = K.image_data_format()
        self.data_format = value.lower()
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):

        if self.data_format == 'channels_first':
            return s[0], s[1], s[2] + 2 * self.padding[0], s[3] + 2 * self.padding[1]
        elif self.data_format == 'channels_last':
            return s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3]

    def call(self, x):

        w_pad, h_pad = self.padding
        if self.data_format == 'channels_first':
            pattern = [[0, 0], [0, 0], [h_pad, h_pad], [w_pad, w_pad]]
        elif self.data_format == 'channels_last':
            pattern = [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]]

        return tf.pad(x, pattern, 'REFLECT')


def ResBlock(input, filters, kernel_size=(3, 3), strides=(1, 1)):
    x = ReflectionPadding2D((1, 1))(input)
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = ReflectionPadding2D((1, 1))(x)
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    out = Add()([input, x])
    return out

class DCGAN:

    @staticmethod
    def build_generator(image_shape, num_gen_filter, num_resblock):

        inputs = Input(shape=image_shape)

        x = ReflectionPadding2D((3, 3))(inputs)
        x = Conv2D(filters=num_gen_filter, kernel_size=(7, 7), padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        n_downsample = 2
        for i in range(n_downsample):
            mul = 2**i
            x = Conv2D(filters=num_gen_filter * mul * 2, kernel_size=(3, 3), strides=2, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

        mul = 2**n_downsample
        for i in range(num_resblock):
            x = ResBlock(x, num_gen_filter * mul)

        for i in range(n_downsample):
            mul = 2**(n_downsample - i)
            x = UpSampling2D()(x)
            x = Conv2D(filters=int(num_gen_filter * mul / 2), kernel_size=(3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

        x = ReflectionPadding2D((3, 3))(x)
        x = Conv2D(filters=3, kernel_size=(7, 7), padding='valid')(x)
        x = Activation('tanh')(x)

        outputs = Add()([x, inputs])
        outputs = Lambda(lambda z: z/2)(outputs)

        model = Model(inputs=inputs, outputs=outputs, name='Generator')
        return model

    @staticmethod
    def build_discriminator(image_shape, num_dis_filter):

        n_layers = 3
        inputs = Input(shape=image_shape)

        x = Conv2D(filters=num_dis_filter, kernel_size=(4, 4), strides=2, padding='same')(inputs)
        x = LeakyReLU(0.2)(x)

        nf_mult, nf_mult_prev = 1, 1
        for n in range(n_layers):
            nf_mult_prev, nf_mult = nf_mult, min(2**n, 8)
            x = Conv2D(filters=num_dis_filter * nf_mult, kernel_size=(4, 4), strides=2, padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(0.2)(x)

        nf_mult_prev, nf_mult = nf_mult, min(2**n_layers, 8)
        x = Conv2D(filters=num_dis_filter * nf_mult, kernel_size=(4, 4), strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(filters=1, kernel_size=(4, 4), strides=1, padding='same')(x)
        x = Flatten()(x)
        x = Dense(1024, activation='tanh')(x)
        x = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=x, name='Discriminator')
        return model
