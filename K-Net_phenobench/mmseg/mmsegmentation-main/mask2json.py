# 导入包
import os
import io
import json
import numpy as np
from pycococreatortools import pycococreatortools
from PIL import Image
import base64
import argparse
import cv2

"""
python mask2json.py -i E:\WY\TianYue_code_c++\gendefect\gendefect\data4\output\small
python mask2json.py -i E:\WY\TianYue_code_c++\gendefect\gendefect\data4\output\small
"""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='CP379.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',default='E:\WY\Huatian\\test2',
                        help='filenames of input images', required=False)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',default='CP379.pth',
                        help='filenames of ouput images')
    parser.add_argument('--cpu', '-c', action='store_true',
                        help="Do not use the cuda version of the net",
                        default=False)
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--no-crf', '-r', action='store_true',
                        help="Do not use dense CRF postprocessing",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=1)

    return parser.parse_args()

def img_tobyte(img_pil):
    '''
    该函数用于将图像转化为base64字符类型
    :param img_pil: Image类型
    :return base64_string: 字符串
    '''
    ENCODING = 'utf-8'
    img_byte = io.BytesIO()
    img_pil.save(img_byte, format='PNG')
    binary_str2 = img_byte.getvalue()
    imageData = base64.b64encode(binary_str2)
    base64_string = imageData.decode(ENCODING)
    return base64_string

if __name__ == "__main__":
    args = get_args()
    # 定义路径
    # ROOT_DIR = 'E:\WY\TianYue\classified\qiege2\gendefects\\0bai\\test\out\\small' # 请输入你文件的根目录
    # ROOT_DIR = args.input[0]
    ROOT_DIR = args.input
    print(ROOT_DIR)
    # Image_DIR = os.path.join(ROOT_DIR, "image") # 目录底下包含图片
    # Label_DIR = os.path.join(ROOT_DIR, "mask") # 目录底下包含label文件

    Image_DIR = r"/home/wangyi/wy/knet/PhenoBench/val/images"
    Label_DIR = r"/home/wangyi/wy/knet/PhenoBench/val/semantics"
    # 读取路径下的掩码
    Label_files = os.listdir(Label_DIR)
    # 指定png中index中对应的label
    class_names = ['_background_', 'basketball', 'person',"3","4"] # 分别表示label标注图中1对应basketball，2对应person。
    for Label_filename in Label_files:
        if not Label_filename.endswith(".png"):  # png
        # if not Label_filename.endswith(".jpg"):  # png
            continue
        # 创建一个json文件
        Json_output = {
            "version": "3.16.7",
            "flags": {},
            "fillColor": [255, 0, 0, 128],
            "lineColor": [0, 255, 0, 128],
            "imagePath": {},
            "shapes": [],
            "imageData": {}}
        print(Label_filename)

        # name = Label_filename.split('.', 3)[0]
        maskname,file_extension  = os.path.splitext(Label_filename)

        imgname = maskname + '.png'
        Json_output["imagePath"] = imgname
        # 打开原图并将其转化为labelme json格式
        imgpath = os.path.join(Image_DIR,imgname)
        # imgpath = os.path.join(Label_DIR,imgname) #  掉包
        image = Image.open(imgpath)
        imageData = img_tobyte(image)
        Json_output["imageData"] = imageData
        # 获得注释的掩码
        maskpath = os.path.join(Label_DIR,Label_filename)
        print("maks",maskpath)
        # binary_mask = np.asarray(np.array(Image.open(maskpath))
        #                          ).astype(np.uint8)

        binary_mask = cv2.imread(maskpath,-1)
        # print(binary_mask.shape)  #
        print("binary_mask.dtype",binary_mask.dtype)  #
        # _, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)
        # cv2.imwrite("E:\WY\TianYue\classified\qiege2\gendefects\\0bai\out\out6\small\mask\\1.jpg", binary_mask)

        # 分别对掩码中的label结果绘制边界点
        for i in np.unique(binary_mask):
            # print(i)
            # if i == 255:
            if i != 0:
                temp_mask = np.where(binary_mask == i, 1, 0)
                # temp_mask2 = temp_mask*255
                # print(temp_mask.shape)  # (1728, 2336, 3)fail  (7000, 9344) ok
                segmentation = pycococreatortools.binary_mask_to_polygon(temp_mask, tolerance=2) # tolerancec参数控制无误差
                # cv2.imwrite("E:\WY\TianYue\classified\qiege2\gendefects\\0bai\out\out6\small\mask\\2.jpg", binary_mask)
                # print(len(segmentation))
                for item in segmentation:
                    if (len(item) > 10):  # 要除以2  10
                        list1 = []
                        for j in range(0, len(item), 2):
                            list1.append([item[j], item[j + 1]])
                        # label = class_names[i]  #
                        label = "lalala" #
                        seg_info = {'points': list1, "fill_color": None, "line_color": None, "label": label,
                                    "shape_type": "polygon", "flags": {}}
                        Json_output["shapes"].append(seg_info)
        Json_output["imageHeight"] = binary_mask.shape[0]
        Json_output["imageWidth"] = binary_mask.shape[1]
        # 保存在根目录下的json文件中
        full_path = '{}/' + maskname + '.json'
        with open(full_path.format(Image_DIR), 'w') as output_json_file:
            json.dump(Json_output, output_json_file)
