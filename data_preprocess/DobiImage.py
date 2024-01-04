import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.signal import savgol_filter
from DobiImage_Utils import DubiImage_Utils as utils
import pydicom
from PIL import Image
import cv2
from tool_code.matlab_python.comfortproc import comfortproc
from tool_code.matlab_python.file2ds import Read_info_from_dcm


class DubiImage:

    def __init__(self, file_dcm, flash_length=23):
        # Ds = pydicom.dcmread(file_dcm)
        answer = Read_info_from_dcm(file_dcm)
        self.totalTime = answer['MainPressionTime']
        answer, answer2, strarr, strarr2 = comfortproc(answer)
        if answer is None:
            print("Error: " + file_dcm + " is Error")
            return
        self.file_dcm = file_dcm
        #self.mask_path = mask_path
        self.Information = answer
        self.num_light = answer['LEDPeriod']
        # success = self.__decoding()
        # self.wrong_file = None
        # if success != 0 or 0 in answer['LED_level']:
        #     self.wrong_file = file_dcm.split('/')[-1]
        #     return
        # image = utils.get_image(Ds, int(answer["flush"]))
        self.DarkImage = self.Information["Dark_image"]
        self.MainImage = self.Information["Main_image"]
        # self.ana_img()
        # self.mask_maker()
        self.MainImage2 = None
        if type(answer2) != type([]):
            self.DarkImage2 = answer2["Dark_image"]
            self.MainImage2 = answer2["Main_image"]
            # self.MainImage = utils.double_process(self.MainImage, self.MainImage2)
        # print("make a normalization image...")
        self.normalImage = utils.normalization(self.MainImage)
        # print("make a mask image...")
        self.imgs = []
        self.imgs_fitter_this = []
        self.imgs_fitter_first = []
        self.flash_length = flash_length

    def getMainImgae2AndNum_light(self):
        return self.MainImage2,self.num_light

    def ana_img(self):
        for idx, img in enumerate(self.MainImage):
            if idx == self.num_light:
                break
            plt.imshow(img, cmap='gray');
            plt.annotate(text='P2', xy=(self.Information['MarkerP2'][0], self.Information['MarkerP2'][1]),
                         xytext=(self.Information['MarkerP2'][0] + 10, self.Information['MarkerP2'][0] + 10),
                         weight='bold', color='r', arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='c'));
            plt.show(block=True)
            plt.close()

            plt.plot(self.MainImage[idx][int(self.Information['MarkerP2'][1]), :]);
            plt.annotate(text='P2', xy=((self.Information['MarkerP2'][0]), self.MainImage[idx, int(self.Information['MarkerP2'][1]), int(self.Information['MarkerP2'][0])]),
                        xytext=(self.Information['MarkerP2'][0] + 10, self.MainImage[idx, int(self.Information['MarkerP2'][1]), int(self.Information['MarkerP2'][0])] + 10),
                        weight='bold', color='r',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='c'));
            plt.show(block=True)
            plt.close()


    def fill_ndarray(self, t1):  # 定义一个函数，把数组中为零的元素替换为一列的均值
        for i in range(t1.shape[1]):
            temp_col = t1[:, i]  # 取出当前列
            nan_num = np.count_nonzero(temp_col != temp_col)  # 判断当前列中是否含nan值
            if nan_num != 0:
                temp_not_nan_col = temp_col[temp_col == temp_col]
                temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()  # 用其余元素的均值填充nan所在位置
        return t1

    def __decoding(self):
        level = self.Information["ILState"]
        light = level[:, -1]
        self.Level = level[:, 4]
        self.num_light = len(np.unique(light[np.where(light != 0)]))  # 灯数量
        max_press = np.max(self.Information["MainPressure"])
        check_wave = len(light)
        # 指针整体减1
        if check_wave > 300:  # 双波段
            print("双波段", end='\t')
            if max_press > 140:
                print("15mm", end='\t')
                if self.num_light == 3 or self.num_light == 4:
                    print("3LED", end='\n')
                    self.timecut = [6, 36, 105, 144]  # 23 13
                    return 0
                elif self.num_light == 5 or self.num_light == 6:
                    print("5LED", end='\n')
                    self.timecut = [10, 40, 105, 145]  # 13 8
                    return 0
                else:
                    return 1
            else:
                print("10mm", end='\t')
                if self.num_light == 3 or self.num_light == 4:
                    print("3LED", end='\n')
                    self.timecut = [6, 36, 105, 144]  # 23 13
                    return 0
                elif self.num_light == 5 or self.num_light == 6:
                    print("5LED", end='\n')
                    self.timecut = [25, 40, 105, 145]  # 13 8 = 21   18
                    return 0
                else:
                    return 1
        else:
            print("单波段", end='\t')
            if max_press > 140:  # 高压
                print("15mm", end='\t')
                if self.num_light == 3 or self.num_light == 4:
                    print("3LED", end='\n')
                    self.timecut = [18, 48, 117, 156]  # 23  13
                    return 0
                elif self.num_light == 5 or self.num_light == 6:
                    print("5LED", end='\n')
                    self.timecut = [25, 55, 120, 160]  # 13 8
                    return 0
                else:
                    return -1
            else:
                print("10mm", end='\t')
                if self.num_light == 3 or self.num_light == 4:
                    print("3LED", end='\n')
                    self.timecut = [18, 48, 117, 156]  # 23  13
                    return 0
                elif self.num_light == 5 or self.num_light == 6:
                    print("5LED", end='\n')
                    self.timecut = [10, 55, 120, 160]  # 13 8
                    return 0
                else:
                    return -1
        return 0

    def fitter(self, image, windows=21, level=1):
        fiter1 = []

        for i in range(image.shape[1]):
            y = np.array(image[:, i])
            yval = savgol_filter(y, windows, level, mode='interp')  # 15 2
            fiter1.append(yval)
        fiter1 = np.array(fiter1).T
        fiter2 = []
        for i in range(image.shape[0]):
            y = np.array(image[i, :])
            yval = savgol_filter(y, windows, level, mode='interp')  # 15 2
            fiter2.append(yval)
        fiter2 = np.array(fiter2)
        fiter = np.hypot(fiter1, fiter2)
        return fiter

    def make_the_main_images(self, type_flag, name):
        # first_map = self.__mergeImage(start - self.num_light, start) * self.mask
        # first_map = self.fitter(first_map) * self.mask
        start, end = 0, self.MainImage.shape[0]
        first_map = self.__mergeImage(start, 0 + self.num_light)
        # first_map = self.fitter(first_map) * self.mask
        for i in np.arange(start, end, self.num_light):
            this_i_map = self.__mergeImage(i, i + self.num_light)  # - dark
            # fitter_this = self.fitter(this_i_map)
            # Grey_sp_shows1 = (this_i_map - fitter_this) * self.mask   # -befor
            Grey_sp_shows1 = this_i_map * self.mask
            Grey_sp_shows2 = (this_i_map - first_map) * self.mask
            self.imgs_fitter_this.append(Grey_sp_shows1)
            self.imgs_fitter_first.append(Grey_sp_shows2)

    def make_the_main_images_v2(self, type_flag, name):
        start, end = 0, self.MainImage.shape[0]
        first_map = self.__mergeImage_wave(start, 0 + self.num_light) * self.mask
        # first_map = self.fitter(first_map) * self.mask
        for i in np.arange(start, end, self.num_light):
            this_i_map = self.__mergeImage_wave(i, i + self.num_light) * self.mask  # - dark
            # fitter_this = self.fitter(this_i_map)
            # Grey_sp_shows1 = (this_i_map - fitter_this) * self.mask   # -befor
            Grey_sp_shows1 = this_i_map * self.mask
            Grey_sp_shows2 = (this_i_map - first_map) * self.mask
            first_map = this_i_map
            self.imgs_fitter_this.append(Grey_sp_shows1)
            self.imgs_fitter_first.append(Grey_sp_shows2)

    def check_p2Jump(self, light_number):
        ind, col = int(self.Information['MarkerP2'][1]), int(self.Information['MarkerP2'][0])
        l_re, r_re = np.min(np.where(self.mask[ind, :] != 0)), np.max(np.where(self.mask[ind, :] != 0))
        width = r_re - l_re
        if self.num_light == 3:
            if light_number % 3 == 0:
                if (col - l_re) / width > 0.66:
                    return True
                else:
                    return False
            elif light_number % 3 == 1:
                return False
            elif light_number % 3 == 2:
                if (col - l_re) / width > 0.33:
                    return False
                else:
                    return True
        else:
            if light_number % 5 == 0:
                if (col - l_re) / width > 0.8:
                    return True
                else:
                    return False
            elif light_number % 5 == 1:
                if (col - l_re) / width > 0.6:
                    return True
                else:
                    return False
            elif light_number % 5 == 2:
                return False
            elif light_number % 5 == 3:
                if (col - l_re) / width > 0.6:
                    return False
                else:
                    return True
            elif light_number % 5 == 4:
                if (col - l_re) / width > 0.8:
                    return False
                else:
                    return True
    def set_used(self, used_m):
        used = []
        if used_m is None:
            for i in range(self.MainImage.shape[0]):
                if self. check_p2Jump(i):
                    continue
                else:
                    used.append(i)
        else:
            flames = np.array(used_m[0].split(',')).astype(int)
            for i in range(self.MainImage.shape[0]):
                if i % self.num_light in flames:
                    used.append(i)
        return np.array(used)

    def cut_segment_for_mask(self, start, end):
        # mask 生成专用
        # 没有遮罩直接合成
        # self.imgs 在mask_maker 函数中会被销毁
        for i in np.arange(start, end, self.num_light):
            this_i_map = self.__mergeImage_without_filter(i, i + self.num_light)  # - dark
            show = this_i_map
            self.imgs.append(show)

    def __mergeImage_without_filter(self, start, end):
        image = np.zeros([self.MainImage.shape[1], self.MainImage.shape[2]])
        maxs = []
        for i in range(start, end):
            maxs.append(np.max(self.MainImage[i]))  # *led_kernel
            single = (self.normalImage[i])  # 权重 乘 像素# * led_kernel* kernal
            image += single  # 差值越大，原图吸收越明显，即数字越大吸收越明显
        image = image * np.mean(maxs)
        return image

    def __multi_mergeImage(self, start, end):
        image = np.zeros([102, self.MainImage.shape[2]]).astype(np.uint8)
        # 读取灰度图像
        for i in range(start, end):
            img = (self.normalImage[i].copy() * 255 * self.mask).astype(np.uint8)
            # img = utils.denoise_mask_median(img, self.mask)
            img = utils.denoise_mask_bilateral_filter(img, self.mask)
            resize_part = np.zeros([26, 128])
            img = np.row_stack((img, resize_part)).astype(np.uint8)
            image = np.row_stack((image, resize_part)).astype(np.uint8)
            level = 6
            G1= img.copy()
            G2 = image.copy()
            gp1 = [G1]
            gp2 = [G2]
            for idx in range(6):
                G1 = cv2.pyrDown(G1)
                G2 = cv2.pyrDown(G2)
                gp1.append(G1)
                gp2.append(G2)

            lp1 = [gp1[level]]
            lp2 = [gp2[level]]
            for j in range(level, 0, -1):
                GE1 = cv2.pyrUp(gp1[j], dstsize=(gp1[j - 1].shape[1], gp1[j - 1].shape[0]))
                GE2 = cv2.pyrUp(gp2[j], dstsize=(gp2[j - 1].shape[1], gp2[j - 1].shape[0]))
                L1 = cv2.subtract(gp1[j - 1], GE1)
                L2 = cv2.subtract(gp2[j - 1], GE2)
                lp1.append(L1)
                lp2.append(L2)

            # 融合拉普拉斯金字塔
            LS = []
            for la, lb in zip(lp1, lp2):
                rows, cols = la.shape
                ls = np.hstack((la[:, :cols // 2], lb[:, cols // 2:]))
                LS.append(ls)

            # 重建图像
            ls_ = LS[0]
            for j in range(1, level + 1):
                ls_ = cv2.pyrUp(ls_, dstsize=(LS[j].shape[1], LS[j].shape[0]))
                ls_ = cv2.add(ls_, LS[j])

            # 保存融合后的图像
            image = ((ls_).astype(np.uint8)[0:102, :] * self.mask).astype(np.uint8)
            image = utils.clahe_resize(image, self.mask, clipLimit=5, tileGridSize=(2, 2)).astype(np.uint8)

        return image.astype(float)

    def __mergeImage(self, start, end):
        image = np.zeros([self.MainImage.shape[1], self.MainImage.shape[2]])
        maxs = []
        for i in range(start+1, end):
            # if self.check_p2Jump(i):
            #     continue
            maxs.append(np.max(self.MainImage[i]))  # *led_kernel
            # if self.MainImage[i][int(self.Information['MarkerP2'][0])][int(self.Information['MarkerP2'][1])]:
            single = (self.normalImage[i])  # 权重 乘 像素# * led_kernel* kernal
            image += single  # 差值越大，原图吸收越明显，即数字越大吸收越明显 #filter_img-
            # image += self.MainImage[i]
            # image += self.normalImage[i]
        # image = self.fill_ndarray(image)
        image = image * np.mean(maxs)
        return image

    def __mergeImage_wave(self, start, end):

        image = np.zeros([self.MainImage.shape[1], self.MainImage.shape[2]])
        maxs = []
        for i in range(start, end):
            # if self.check_p2Jump(i):
            #     continue
            # self.pca_merge2wave(self.MainImage[i], self.MainImage2[i])
            img1 = self.MainImage[i] - self.MainImage2[i]
            img2 = self.normalImage[i] - self.normalImage2[i]
            # image += img1
            image += img2
            maxs.append(np.max(img1))
        image *= np.mean(maxs)
        return image

    def pca_merge2wave(self, img1, img2):
        import cv2 as cv
        from sklearn.decomposition import PCA

        # img1_gray = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
        # img2_gray = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)

        pca_1 = PCA(n_components=1)
        reduced_x1 = pca_1.fit_transform(img1)
        pca_2 = PCA(n_components=1)
        reduced_x2 = pca_2.fit_transform(img2)
        reduced_x1[:, 0] = reduced_x2[:, 0] + 1 * reduced_x1[:, 0]
        reduced_x = np.dot(reduced_x1, pca_1.components_) + pca_1.mean_

        plt.imshow(reduced_x)
        plt.show()

    def back_cut(self, back, size=4):
        for j in range(back.shape[1]):
            back_process_col = back[:, j]
            idx = np.argwhere(back_process_col != 0).reshape(-1)
            if len(idx) == 0:
                continue
            elif len(idx) > 0:
                cut = idx[0] + size
                back[cut - size:cut, j] = 0
        for i in range(back.shape[0]):
            back_process_row = back[i, :]
            idy = np.argwhere(back_process_row != 0).reshape(-1)
            if len(idy) == 0:
                continue
            elif len(idy) > 0:
                cut = idy[0] + size
                back[i, cut - size:cut] = 0
                cut = idy[-1]
                back[i, cut:cut + size] = 0
        return back

    # def read_info_from_Ds(self, Ds):
    #     answer = {}
    #     DICOMRev = Ds.get_item([0x5000, 0x2500]).value
    #     if type(DICOMRev) != type("DOBI IMAGE 2"):
    #         DICOMRev = DICOMRev.decode()
    #     if DICOMRev == "DOBI IMAGE 2":
    #         Step = 20
    #         dn = 1
    #     elif DICOMRev == "DOBI IMAGE 1":
    #         Step = 19
    #         dn = 0
    #     else:
    #         return None
    #     Patient_Comments = Ds.get_item([0x0010, 0x4000]).value
    #     try:
    #         Age = Ds.get_item([0x0010, 0x1010]).value[1:3]
    #         Age = int(Age)
    #     except:
    #         Age = -1
    #     Marks = Ds.get_item([0x5003, 0x3000]).value
    #     Marks = utils.uncoding_mark(Marks)
    #     MarkerP1 = [float(Marks[0] * 256 + Marks[1]), float(Marks[2] * 256 + Marks[3])];
    #     MarkerP2 = [float(Marks[4] * 256 + Marks[5]), float(Marks[6] * 256 + Marks[7])];
    #     CurveData_1 = Ds.get_item([0x5000, 0x3000]).value
    #     CurveData_1 = utils.uncoding_mark(CurveData_1)
    #     CurveData_2 = Ds.get_item([0x5002, 0x3000]).value
    #     CurveData_2 = utils.uncoding_mark(CurveData_2)
    #     NFr = len(CurveData_2) / Step;
    #     CurveData_2 = CurveData_2.reshape([int(NFr), int(Step)])
    #     MainPressure = CurveData_2[:, dn + 17]
    #     ILState = CurveData_2[:, dn + 5:dn + 16]
    #     if DICOMRev == "DOBI IMAGE 2":
    #         MainTime = 0.001 * (2 ^ 16 * CurveData_2[:, 0] + CurveData_2[:, 1])
    #     elif DICOMRev == "DOBI IMAGE 1":  # need debug
    #         index = np.arange(0, len(CurveData_1), 2)
    #         MainTime = 0.001 * (CurveData_1[index])
    #     else:
    #         return None
    #     LED_level = utils.check_LED_level(ILState)
    #     Laterality = Ds.get_item([0x0020, 0x0060]).value
    #     flush = Ds.get_item([0x0028, 0x0008]).value
    #     Rows = Ds.get_item([0x0028, 0x0010]).value
    #     Rows = utils.uncoding_mark(Rows)
    #     Columns = Ds.get_item([0x0028, 0x0011]).value
    #     Columns = utils.uncoding_mark(Columns)
    #     answer.update({"DICOMRev": DICOMRev})
    #     answer.update({"Patient_Comments": Patient_Comments.decode()})
    #     answer.update({"Laterality": Laterality.decode()})
    #     answer.update({"flush": flush.decode()})
    #     answer.update({"Rows": int(Rows)})
    #     answer.update({"Columns": int(Columns)})
    #     answer.update({"MainPressionTime": MainTime})
    #     answer.update({"MainPressure": MainPressure})
    #     answer.update({"ILState": ILState})
    #     answer.update({"MarkerP1": MarkerP1})
    #     answer.update({"MarkerP2": MarkerP2})
    #     answer.update({"LED_level": LED_level})
    #     answer.update({"Age": Age})
    #     return answer

    def mask_maker(self):
        # 生成一个mask
        # 删除
        try:
            mask = self.mask_path
            if type(mask) != type(np.array(range(1))) and mask != None:
                # mask = np.array(mask)
                mask = np.array(Image.open(mask).convert('L'))
            if len(mask.shape) == 3:
                back = mask[:, :, 0]
                mask = (back - np.min(back)) / (np.max(back) - np.min(back))
            if np.max(mask) > 1:
                # mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
                mask = np.where(mask != np.min(mask), 1, 0)
            self.mask = mask.astype(np.uint8)
            return
        except Exception as e:
            print("Warning: No mask file can be found, make a mask.")
        mask = np.ones([102, 128])
        self.cut_segment_for_mask(0, 0 + self.num_light)
        self.cut_segment_for_mask(self.timecut[2], self.timecut[3])
        dark = self.imgs[0]  # np.min(dbimage.DarkImage, axis=0)
        for img in self.imgs:
            img = img - dark
            img = self.fitter(img, 21, 1)
            if np.min(img) == np.max(img):
                continue
            img = img - np.min(img)
            tmp = utils.threshold_By_OTSU(img)
            tmp = utils.imfill(tmp)
            tmp = ndimage.median_filter(tmp, (5, 5))
            mask *= tmp
        mask = np.where(mask > 0, 1, 0)
        for i in range(mask.shape[1]):
            line = mask[:, i]
            index = np.argwhere(line > 0)
            if len(index) <= 0:
                continue
            start = np.min(index)
            line[start:start + 7] = 0
            mask[:, i] = line
        del self.imgs
        self.mask = mask.astype(np.uint8)
        self.mask = self.back_cut(self.mask)

    def output_format_images(self):
        # 按照一个 this 一个 first 插入图片
        tmp_first = np.array(self.imgs_fitter_first)
        tmp_this = np.array(self.imgs_fitter_this)
        image_this = []
        image_first = []
        if tmp_first.shape[0] < self.flash_length:
            if self.flash_length == 36:  # 21 to 36
                print("information: 5LED just has 21 flush, change the shape to 36")
                for i in range(0, 3):
                    image_this.append(tmp_this[i])
                    image_first.append(tmp_first[i])
                for i in range(3, len(tmp_this) - 3):
                    image_this.append(tmp_this[i])
                    image_first.append(tmp_first[i])
                    image_this.append(tmp_this[i])
                    image_first.append(tmp_first[i])
                for j in range(0, 3):
                    image_this.append(tmp_this[i + j])
                    image_first.append(tmp_first[i + j])
            elif self.flash_length == 46:  # 27 to 46
                print("information: 5LED just has 27 flush, change the shape to 46")
                for i in range(0, 4):
                    image_this.append(tmp_this[i])
                    image_first.append(tmp_first[i])
                for i in range(4, len(tmp_this) - 4):
                    image_this.append(tmp_this[i])
                    image_first.append(tmp_first[i])
                    image_this.append(tmp_this[i])
                    image_first.append(tmp_first[i])
                for j in range(0, 4):
                    image_this.append(tmp_this[i + j])
                    image_first.append(tmp_first[i + j])
            else:  # 13 to 23
                print("information: 5LED just has 13 flush, change the shape to 23")
                for i in range(0, 2):
                    image_this.append(tmp_this[i])
                    image_first.append(tmp_first[i])
                for i in range(2, len(tmp_this) - 1):
                    image_this.append(tmp_this[i])
                    image_first.append(tmp_first[i])
                    image_this.append(tmp_this[i])
                    image_first.append(tmp_first[i])
                for j in range(0, 1):
                    image_this.append(tmp_this[i + j])
                    image_first.append(tmp_first[i + j])
        elif tmp_first.shape[0] > self.flash_length:
            idxs = [0, len(tmp_this) // 2, -1]
            image_this = list(tmp_this[idxs])
            image_first = list(tmp_this[idxs])
        else:
            for i in range(len(tmp_this)):
                image_this.append(tmp_this[i])
                image_first.append(tmp_first[i])
        if len(image_this) != len(image_first) != self.flash_length:
            print(str(self.file_dcm) + "flash is error")
            return [], []
        # else:
        # print("out put image shape is "+ str(len(image_first)))
        # print()
        return image_this, image_first

    def output_format_images_v2(self):
        # 按照一个 this 一个 first 插入图片
        tmp_first = np.array(self.imgs_fitter_first)
        tmp_this = np.array(self.imgs_fitter_this)
        image_this = []
        image_first = []

        if tmp_first.shape[0] == self.flash_length:
            if self.flash_length == 36:  # 21 to 36
                print("information: 3LED just has 36 flush, change the shape to 21")
                for i in range(0, 21, 2):
                    image_this.append(tmp_this[i])
                    image_first.append(tmp_first[i])
                for i in range(21, 23, 1):
                    image_this.append(tmp_this[i])
                    image_first.append(tmp_first[i])
                for i in range(23, 34, 2):
                    image_this.append(tmp_this[i])
                    image_first.append(tmp_first[i])
                for i in range(34, 36, 1):
                    image_this.append(tmp_this[i])
                    image_first.append(tmp_first[i])
            elif self.flash_length == 46:  # 27 to 46
                print("information: 5LED just has 27 flush, change the shape to 46")
                for i in range(0, 4):
                    image_this.append(tmp_this[i])
                    image_first.append(tmp_first[i])
                for i in range(4, len(tmp_this) - 4):
                    image_this.append(tmp_this[i])
                    image_first.append(tmp_first[i])
                    image_this.append(tmp_this[i])
                    image_first.append(tmp_first[i])
                for j in range(0, 4):
                    image_this.append(tmp_this[i + j])
                    image_first.append(tmp_first[i + j])
            else:  # 13 to 23
                print("information: 5LED just has 13 flush, change the shape to 23")
                for i in range(0, 2):
                    image_this.append(tmp_this[i])
                    image_first.append(tmp_first[i])
                for i in range(2, len(tmp_this) - 1):
                    image_this.append(tmp_this[i])
                    image_first.append(tmp_first[i])
                    image_this.append(tmp_this[i])
                    image_first.append(tmp_first[i])
                for j in range(0, 1):
                    image_this.append(tmp_this[i + j])
                    image_first.append(tmp_first[i + j])
        else:
            for i in range(len(tmp_this)):
                image_this.append(tmp_this[i])
                image_first.append(tmp_first[i])
        if len(image_this) != len(image_first) != 21:
            print(str(self.file_dcm) + "flash is error")
            return [], []
        # else:
        # print("out put image shape is "+ str(len(image_first)))
        # print()
        return image_this, image_first
