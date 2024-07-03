import cv2
import numpy as np
import os


def convert_image_to_int8_and_save(input_path, output_path):
    # 读取图像
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    # 检查图像是否成功读取
    if image is None:
        print(f"无法读取图片：{input_path}")
        return

    # 打印图像数据类型和形状
    print(f"原始图像数据类型: {image.dtype}")
    print(f"原始图像形状: {image.shape}")

    # 将图像转换为int16类型
    image_int16 = image.astype(np.int16)

    # 打印转换后的数据类型
    print(f"转换后的图像数据类型（int16）: {image_int16.dtype}")

    # 将图像转换为int8类型
    image_int8 = image_int16.astype(np.int8)

    # 打印转换后的数据类型
    print(f"转换后的图像数据类型（int8）: {image_int8.dtype}")

    # 转换为灰度图
    if len(image_int8.shape) == 3:
        image_gray = cv2.cvtColor(image_int8, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image_int8

    # 保存灰度图像
    cv2.imwrite(output_path, image_gray)
    print(f"图像已保存为灰度图：{output_path}")


def process_images_in_folder(input_folder, output_folder):
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # 只处理图像文件（根据需要可以添加更多图像扩展名）
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            output_path = os.path.join(output_folder, filename)
            convert_image_to_int8_and_save(input_path, output_path)


# 使用示例
input_folder_path = '/disk2/wangyidata/other_datasets/ADEChallengeData2016_plane/annotations/validation'  # 替换为你的输入文件夹路径
output_folder_path = '/disk2/wangyidata/other_datasets/ADEChallengeData2016_plane/annotations/validation2'  # 替换为你的输出文件夹路径
process_images_in_folder(input_folder_path, output_folder_path)
