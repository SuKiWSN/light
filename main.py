import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def get_binary_image(gray_image, index):
    aug_image = (255 * ((gray_image / 255) ** index)).astype(np.uint8)
    _, binary_image = cv2.threshold(aug_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # dilated_image_ellipse = cv2.dilate(binary_image, kernel_ellipse, iterations=0)
    # return dilated_image_ellipse
    return binary_image


def save_figure(binary_image, savePath):
    plt.imshow(binary_image)
    image_name = os.path.basename(savePath)
    plt.title(image_name)
    plt.axis("off")
    plt.savefig(savePath)


def adjust_figure(image, gamma=2.4):
    aug_img = np.power(image / float(np.max(image)), gamma)
    aug_img = aug_img * 255
    aug_img = aug_img.astype(np.uint8)
    # plt.imshow(aug_img)
    # plt.savefig("/Users/suki/Downloads/11.jpg")
    # exit()
    return aug_img


def calculate_background_brightness(image_path):
    # 读取彩色图像
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    image_name = os.path.basename(image_path)
    aug_img = adjust_figure(image)
    # save_figure(aug_img, os.path.join("/Users/suki/Downloads/aug", image_name))
    aug_img = cv2.cvtColor(aug_img, cv2.COLOR_BGR2GRAY)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # 将彩色图像转换为灰度图像
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, light_binary_image = cv2.threshold(gray_image, 254, 255, cv2.THRESH_BINARY)
    _, light_binary_image2 = cv2.threshold(gray_image, 246, 255, cv2.THRESH_BINARY)
    light_binary_image = cv2.dilate(light_binary_image, kernel_ellipse, iterations=5)
    # light_binary_image2 = cv2.dilate(light_binary_image2, kernel_ellipse, iterations=20)
    # light_binary_image = get_binary_image(gray_image, 20)
    light_mask = light_binary_image == 255
    light_mask2 = light_binary_image2 == 255
    light_proportion = np.sum(light_mask) / np.sum(light_mask2)
    # light_proportion = np.power(np.sum(light_mask) / np.sum(light_mask2), 0.1)
    # light_proportion = np.sum(light_mask) / (w*h)
    # light_mean_luminance = np.mean(gray_image[light_mask])
    light_mean_luminance = np.mean(aug_img[light_mask])
    print(light_mean_luminance)
    # save_figure(light_binary_image, os.path.join(segLightDir, image_name))
    _, light_binary_image3 = cv2.threshold(gray_image, 253, 255, cv2.THRESH_BINARY)
    light_mask3 = light_binary_image3 == 255
    gray_image[light_mask3] = 0
    # bg_binary_image = get_binary_image(gray_image, 0.82)
    _, bg_binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # _, bg_binary_image = cv2.threshold(gray_image, 235, 255, cv2.THRESH_BINARY)
    # bg_binary_image -= light_binary_image
    bg_mask = bg_binary_image == 255
    # bg_mean_luminance = np.mean(gray_image[bg_mask])
    bg_mean_luminance = np.mean(aug_img[bg_mask])
    # save_figure(bg_binary_image, os.path.join(segBgDir, image_name))
    grade = 8 * np.log10(0.25/bg_mean_luminance * light_proportion*1000 * light_mean_luminance**2)
    return grade


def grade2label(grade, thresh):
    if grade < thresh[0]:
        return 1
    elif thresh[0] <= grade < thresh[1]:
        return 2
    else:
        return 3


def main(imagePath):
    background_brightness = calculate_background_brightness(imagePath)
    thresh = [36, 40]
    label = grade2label(background_brightness, thresh)
    return label


if __name__ == '__main__':
    # 示例图像路径
    # imgRoot = "/Users/suki/Downloads/UGRpicture-0527"
    imgRoot = "/Users/suki/Downloads/20240531/"
    # imgRoot = "/Users/suki/Downloads/croped_ugr_img"
    # imgRoot = "/Users/suki/Desktop/testlight"
    saveLightDir = "/Users/suki/Downloads/light"
    saveBgDir = "/Users/suki/Downloads/bg"
    thresh = [35, 40]
    if not os.path.exists(saveLightDir):
        os.makedirs(saveLightDir)
    if not os.path.exists(saveBgDir):
        os.makedirs(saveBgDir)
    imgList = []
    for i in range(1, 22):
        imgList.append(f"{i}.jpg")
    text = open("grades.csv", 'w')
    for imageName in imgList:
        if imageName.split('.')[-1] in ['png', 'jpg']:
            image_path = os.path.join(imgRoot, imageName)
            background_brightness = calculate_background_brightness(image_path)
            text.write(imageName + ", " + str(background_brightness) + "\n")
            print(f'{imageName} 估计的背景亮度: {background_brightness:.2f}')
