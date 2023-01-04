# iFixer
Install the following packages before running the code

1. tensorflow
2. keras
3. tensorflow_hub
4. pytesseract

To install pytesseract on a windows machine, we have provided an .exe file which in present in 'code' folder
Install it in the default path suggested by the installer. Because hard-coded is set to the installed file location.
Otherwise Runtime error can occur. There isn't a way to provide the path while run. 
Harded coded path is ==> "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

We have provided the code to test our model with 10 images located in 'test_images' folder. Text recognition result 
will be printed on to the console. Images from deblurring and super resolution stage will be saved in 'results' folder.

*************************
If faced with an issue while running the code locally. It's highly recommended to run the code on Google Colab.
But before running the code, pytesseract needs to be installed. Use the following commands to do that:

!sudo apt install tesseract-ocr
!pip install pytesseract

Copy all the directory's present in code folder to the root directory of colab. Please replace test.py file
with colab_test.py present under 'colab' folder and use them to test the code. 


Problem statement:
This work is on motion deblurring and super-resolution. Significant progress has been recently achieved in related areas of image super-resolution and in-painting by applying generative adversarial networks (GANs). GANs are known for the ability to preserve texture details in images, create solutions that are close to the real image manifold and look perceptually convincing. Inspired by recent work on image super-resolution and image-to-image translation by generative adversarial networks, we treat deblurring as a special case of such image-to-image translation. For image super-resolution we will be using state-of-the-art model which are readily available.

Dataset:
The IIIT 5K-word dataset is harvested from Google image search. Query words like billboards, signboard, house numbers, house name plates, movie posters were used to collect images. The dataset contains 5000 cropped word images from Scene Texts and born-digital images. The dataset is divided into train and test parts.

Algorithm:
To estimate the blur kernel, we employ GANs for this task. The idea of generative adversarial networks, introduced by Goodfellow et al. [1], is a defined as game between two networks, generator, and discriminator. Usually, generator receives noise as an input and generates samples as output. Whereas discriminator receives real and generated samples as input and tries to distinguish between them.
Deblurring task can be formulated as the following equation:
The goal is here to recover sharp Image, given only blurred image by estimating the blur kernel. Instead of taking random noise as input, Generator takes in the blurred image as input and outputs deburred image (estimated sharp image) and then discriminator takes these generated image and sharp image as input and tries to distinguish between them.
Generator architecture consists of two convolution blocks with stride, 3 residual blocks and 2 transposed convolution blocks. Each residual block consists of convolution layer, normalization layer, reflection padding layer with dropout of 0.5 and ReLU activation function. As for the discriminator network, it’s similar to PatchGAN [2].
As for the loss function, we use formulated it as a combination of Adversarial Loss and content loss. MAE and MSE are two classical choices for content loss but here we use Perpetual loss [3], which is simple L2-loss but based on difference of generated and target image feature maps. As for the adversarial loss WGAN-GP [4], which is more stable and generates higher quality results.
Image Super Resolution is followed by the image deblurring operation in our image processing pipeline. Image Super Resolution is the process of enhancing an image's resolution from low resolution to high resolution. The goal of this ISR architecture is to preserve the finer textures of the image when we upscale it, guaranteeing that its quality is unaffected.
The Image Super Resolution task in our project is accomplished by implementing ESRGAN [5] model, an improved model of Super Resolution Generative Adversarial Networks by enhancing network architecture, adversarial loss and perceptual loss. To create higher resolution images, Super-resolution GAN uses deep network in conjunction with an adversary network.
The RRDB block combines a multi-level residual network and dense connections without Batch Normalization, substitutes the residual block in ESRGAN's baseline ResNet-style architecture. The RRDB block draws inspiration from the DenseNet architecture which directly connects all of the levels in the residual block. We have implemented ESRGAN pre-trained model in our work which was trained on DIV2K dataset.

Results:
To evaluate out results we use SSIM and PSNR as metrics. The Structural Similarity Index (SSIM) is a perceptual metric that quantifies image quality degradation caused by processing such as data compression or by losses in data transmission. It is given by the following formula:
The term peak signal-to-noise ratio (PSNR) is an expression for the ratio between the maximum possible value (power) of a signal and the power of distorting noise that affects the quality of its representation.
We train our Deblurring model for 5 epochs. And following are the results:
We tested are deblurring model on almost 2000 images. We got an average SSIM score of 0.528 with minimum and maximum score of 0.13 and 0.70 respectively, and average PSNR score of 15.36 with minimum and maximum score of 10.10 and 26.6 respectively.
Next, we ran our deblurred images through super resolution model. The model has achieved a PSNR score of 27.293 for the ESRGAN model. We also compared our model with some other algorithms like Nearest Neighbor, Linear and Cubic scaling algorithm by using PyTesseract for text extraction, the correctness of the output from super resolution model was comparatively higher than other algorithms.
From the analysis, the GAN based model achieved overall accuracy better than other resampling techniques for image processing. Even though the model is computationally intensive, the results obtained were relatively better with deblurring and image super-resolution in the image enhancement pipeline.
Figure 1: Deblurred Image
Figure 2: Sharp Image
Since, Text Extraction was only a testing module for our project, we have used simple OCR system like PyTesseract for the task. However, much improved results can be obtained by using other Text Extraction systems that have higher accuracy and reliability than PyTesseract for the Scene Text Recognition tasks.
