from PIL import Image
import os
from natsort import natsorted
import pandas as pd

def convert_and_delete_png(folder_path):
    i=1276
    for filename in natsorted(os.listdir(folder_path)):
        
        # # Đường dẫn đầy đủ của file
        file_path = os.path.join(folder_path, filename)
        new_file_name = f'gen{i}.jpg'
        new_file_path = os.path.join(folder_path, new_file_name)
        print(new_file_path)

        os.rename(file_path, new_file_path)
        i+=1
        # # Mở ảnh .png
        # png_image = Image.open(file_path)
        
        # # Chuyển đổi ảnh sang chế độ RGB
        # rgb_image = png_image.convert('RGB')
        
        # # Tạo tên file mới với đuôi .jpg
        # new_filename = filename.replace(".png", ".jpg")
        
        # # Lưu ảnh với định dạng .jpg
        # rgb_image.save(os.path.join(folder_path, new_filename), "JPEG")
        
        # # Xóa file .png sau khi chuyển đổi
        # os.remove(file_path)
            
        # print(f"Converted {filename} to {new_filename} and deleted the original .png file")
    
def sample(df):
    df = df.sample(n=87)