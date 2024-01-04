import numpy as np
from scipy import ndimage
import cv2

class DubiImage_Utils:

    def uncoding_mark(byte):
        if type(byte) == type(1):
            return byte
        i = 0
        numlist = []
        while i < len(byte):
            a = byte[i + 1]
            b = byte[i]
            num = a * 256 + b
            i += 2
            numlist.append(num)
        return np.array(numlist)

    def check_LED_level(ILState):
        LeveL_A = ILState[:, 4]
        LeveL_B = ILState[:, 5]
        check = np.sum(LeveL_A - LeveL_B)
        if check == 0:
            return LeveL_A
        else:
            print("LED level error")
            return -1

    def get_image(Ds, length):
        image_matrix = Ds.pixel_array
        answer = {}
        Dark_image = image_matrix[length - 2:length]
        Main_image = image_matrix[:length - 2]
        answer.update({"Dark_image": Dark_image})
        answer.update({"Main_image": Main_image})
        return answer

    def normalizatiion_by_flash(image):
        _range = np.max(image) - np.min(image)
        temp_data = (image - np.min(image)) / _range
        return temp_data

    def normalization(MainImage):
        # 归一化
        _range = np.max(MainImage) - np.min(MainImage)
        normalImage = (MainImage-np.min(MainImage)) / _range
        return normalImage

    def chosen_normalization(MainImage, used):
        # 归一化
        _range = np.max(MainImage[used]) - np.min(MainImage[used])
        normalImage = (MainImage-np.min(MainImage[used])) / _range
        return normalImage

    def threshold_By_OTSU(image):
        # image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = np.log(image + 1)
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
        # image = np.where(image>100,100,image)
        gray = image.astype(np.uint8)
        ret, binary = cv2.threshold(gray, 0, 255,
                                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # cv2.THRESH_MASKcv2.THRESH_BINARY |cv2.THRESH_TRUNC|cv2.THRESH_OTSU
        # print("threshold value %s" % ret)  # 打印阈值，超过阈值显示为白色，低于该阈值显示为黑色
        return binary

    def handle_noise(img, flag=-1):
        img = (img * -1).astype(np.float32)
        idx = np.nonzero(img)
        u_idx = np.min(idx[0])
        d_idx = np.max(idx[0])
        l_idx = np.min(idx[1])
        r_idx = np.max(idx[1])
        cnt = len(np.where(img < 0)[0])
        if cnt <= 0.15 * (d_idx - u_idx + 1) * (r_idx - l_idx + 1):
            img = np.where(img< 0, 1, img)
        return img

    def imfill(a):
        return ndimage.binary_fill_holes(a)

    def clahe_resize(img, dot, clipLimit=5, tileGridSize=(2, 2)):
        hist_train_x = img.copy()
        # _range = np.max(img_tmp) - np.min(img_tmp)
        # if _range == 0:
        #     _range = 1
        # hist_train_x = img_tmp - np.min(img_tmp) / _range
        idx = np.nonzero(hist_train_x)
        u_idx = np.min(idx[0])
        d_idx = np.max(idx[0])
        l_idx = np.min(idx[1])
        r_idx = np.max(idx[1])
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        hist_train_x[u_idx: d_idx + 1, l_idx: r_idx + 1] = clahe.apply(hist_train_x[u_idx: d_idx + 1, l_idx: r_idx + 1])
        return hist_train_x * dot

    def denoise_mask_median(img, mask):
        mask_uint8 = mask.copy().astype(np.uint8)
        img_masked = cv2.bitwise_and(img, img, mask=mask_uint8)
        img_masked_denoise = cv2.medianBlur(img_masked, 3)
        img_denoised = cv2.add(img_masked_denoise, cv2.bitwise_not(mask), dtype=cv2.CV_8U)

        return img_denoised

    def denoise_mask_bilateral_filter(img, mask, d=5, sigC=30, sigS=10):
        img_denoise = cv2.bilateralFilter(img, d, sigC, sigS)

        return img_denoise * mask

    def double_process(wave1mg, wave2mg):
        return wave1mg - wave2mg