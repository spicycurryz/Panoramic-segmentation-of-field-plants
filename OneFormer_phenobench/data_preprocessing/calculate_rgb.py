import cv2
import numpy as np
import os

def calculate_mean_std(image_folder):
    pixel_sum = np.zeros(3)
    pixel_squared_sum = np.zeros(3)
    num_pixels = 0
    
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg'))]
    
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pixel_sum += image.sum(axis=(0, 1))
        pixel_squared_sum += (image ** 2).sum(axis=(0, 1))
        num_pixels += image.shape[0] * image.shape[1]
    
    mean = pixel_sum / num_pixels
    mean_of_squares = pixel_squared_sum / num_pixels
    variance = mean_of_squares - (mean ** 2)
    std = np.sqrt(np.maximum(variance, 0))
    
    return mean, std

image_folder = '/root/autodl-fs/PhenoBench/train/images'
mean, std = calculate_mean_std(image_folder)
print('PIXEL_MEAN:', mean)
print('PIXEL_STD:', std)
