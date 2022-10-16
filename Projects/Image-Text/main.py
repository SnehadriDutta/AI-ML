from PIL import Image
from pytesseract import pytesseract
import os
import cv2

path_to_tesseract= r'C:\Program Files\Tesseract-OCR\tesseract.exe'

path_to_img = 'RC/'

pytesseract.tesseract_cmd = path_to_tesseract

texts = []
file = r'C:\Users\hp\Desktop\output.txt'
count = 0

with open(file, "w") as txt_file:
    for root, dirs, file_names in os.walk(path_to_img):
        for file in file_names:
            img = cv2.imread(path_to_img + file)
            scale_percent = 200  # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            text = pytesseract.image_to_string(resized)
            txt_file.write(file + "\n\n" + text + "\n")
            txt_file.write("________________________________________________________________________________\n")
            count += 1
            print(f'Writing image {count}')


print('completed')

