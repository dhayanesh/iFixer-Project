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
