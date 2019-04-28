# -*- coding: utf-8 -*-
'''
ISP class
============================================================================================================= 
include main operators of in-camera processing pipeline (ISP) and some corresponding inverse operators.

Citation: 
---------
@article{Guo2019Cbdnet,
  title={Toward convolutional blind denoising of real photographs},
  author={Guo, Shi and Yan, Zifei and Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
  journal={2019 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}

Author: Shi Guo, 01/04/2019
'''


import numpy as np
import cv2
import os
import h5py
import scipy.io
import math
import skimage
from Demosaicing_malvar2004 import demosaicing_CFA_Bayer_Malvar2004
import random

class ISP:
    def __init__(self, curve_path='./'):
        filename = os.path.join(curve_path, '201_CRF_data.mat')
        CRFs = scipy.io.loadmat(filename)
        self.I = CRFs['I']
        self.B = CRFs['B']
        filename = os.path.join(curve_path, 'dorfCurvesInv.mat')
        inverseCRFs = scipy.io.loadmat(filename)
        self.I_inv = inverseCRFs['invI']
        self.B_inv = inverseCRFs['invB']
        self.xyz2cam_all = np.array([[1.0234,-0.2969,-0.2266,-0.5625,1.6328,-0.0469,-0.0703,0.2188,0.6406]
                            ,[0.4913,-0.0541,-0.0202,-0.613,1.3513,0.2906,-0.1564,0.2151,0.7183]
                            ,[0.838,-0.263,-0.0639,-0.2887,1.0725,0.2496,-0.0627,0.1427,0.5438]
                            ,[0.6596,-0.2079,-0.0562,-0.4782,1.3016,0.1933,-0.097,0.1581,0.5181]]
                                    )

    def ICRF_Map(self, img, index=0):
        invI_temp = self.I_inv[index, :]
        invB_temp = self.B_inv[index, :]
        [w, h, c] = img.shape
        bin = invI_temp.shape[0]
        Size = w*h*c
        tiny_bin = 9.7656e-04
        min_tiny_bin = 0.0039
        temp_img = np.copy(img)
        temp_img = np.reshape(temp_img, (Size))
        for i in range(Size):
            temp = temp_img[i]
            start_bin = 1
            if temp > min_tiny_bin:
                start_bin = math.floor(temp/tiny_bin - 1)
            for b in range(start_bin, bin):
                tempB = invB_temp[b]
                if tempB >= temp:
                    index = b
                    if index > 1:
                        comp1 = tempB - temp
                        comp2 = temp - invB_temp[index-1]
                        if comp2 < comp1:
                            index = index - 1
                    temp_img[i] = invI_temp[index]
                    break
        temp_img = np.reshape(temp_img, (h, w, c))
        return temp_img

    def CRF_Map(self, img, index=0):
        I_temp = self.I[index, :]
        B_temp = self.B[index, :]
        [w, h, c] = img.shape
        bin = I_temp.shape[0]
        Size = w * h * c
        tiny_bin = 9.7656e-04
        min_tiny_bin = 0.0039
        temp_img = np.copy(img)
        temp_img = np.reshape(temp_img, (Size))
        for i in range(Size):
            temp = temp_img[i]
            start_bin = 1
            if temp > min_tiny_bin:
                start_bin = math.floor(temp / tiny_bin - 1)
            for b in range(start_bin, bin):
                tempB = I_temp[b]
                if tempB >= temp:
                    index = b
                    if index > 1:
                        comp1 = tempB - temp
                        comp2 = temp - I_temp[index - 1]
                        if comp2 < comp1:
                            index = index - 1
                    temp_img[i] = B_temp[index]
                    break
        temp_img = np.reshape(temp_img, (h, w, c))
        return temp_img

    def RGB2XYZ(self, img):
        xyz = skimage.color.rgb2xyz(img)
        return xyz

    def XYZ2RGB(self, img):
        rgb = skimage.color.xyz2rgb(img)
        return rgb

    def XYZ2CAM(self, img, M_xyz2cam=0):
        if type(M_xyz2cam) is int:
            cam_index = np.random.random((1, 4))
            cam_index = cam_index / np.sum(cam_index)
            M_xyz2cam = (self.xyz2cam_all[0, :] * cam_index[0, 0] + \
                         self.xyz2cam_all[1, :] * cam_index[0, 1] + \
                         self.xyz2cam_all[2, :] * cam_index[0, 2] + \
                         self.xyz2cam_all[3, :] * cam_index[0, 3] \
                         )
            self.M_xyz2cam = M_xyz2cam

        M_xyz2cam = np.reshape(M_xyz2cam, (3, 3))
        M_xyz2cam = M_xyz2cam / np.tile(np.sum(M_xyz2cam, axis=1), [3, 1]).T
        cam = self.apply_cmatrix(img, M_xyz2cam)
        cam = np.clip(cam, 0, 1)
        return cam

    def CAM2XYZ(self, img, M_xyz2cam=0):
        if type(M_xyz2cam) is int:
            cam_index = np.random.random((1, 4))
            cam_index = cam_index / np.sum(cam_index)
            M_xyz2cam = (self.xyz2cam_all[0, :] * cam_index[0, 0] +
                         self.xyz2cam_all[1, :] * cam_index[0, 1] +
                         self.xyz2cam_all[2, :] * cam_index[0, 2] +
                         self.xyz2cam_all[3, :] * cam_index[0, 3]
                         )
        M_xyz2cam = np.reshape(M_xyz2cam, (3, 3))
        M_xyz2cam = M_xyz2cam / np.tile(np.sum(M_xyz2cam, axis=1), [3, 1]).T
        M_cam2xyz = np.linalg.inv(M_xyz2cam)
        xyz = self.apply_cmatrix(img, M_cam2xyz)
        xyz = np.clip(xyz, 0, 1)
        return xyz

    def apply_cmatrix(self, img, matrix):
        r = (matrix[0, 0] * img[:, :, 0] + matrix[0, 1] * img[:, :, 1]
             + matrix[0, 2] * img[:, :, 2])
        g = (matrix[1, 0] * img[:, :, 0] + matrix[1, 1] * img[:, :, 1]
             + matrix[1, 2] * img[:, :, 2])
        b = (matrix[2, 0] * img[:, :, 0] + matrix[2, 1] * img[:, :, 1]
             + matrix[2, 2] * img[:, :, 2])
        r = np.expand_dims(r, axis=2)
        g = np.expand_dims(g, axis=2)
        b = np.expand_dims(b, axis=2)
        results = np.concatenate((r, g, b), axis=2)
        return results

    def BGR2RGB(self, img):
        b, g, r = cv2.split(img)
        rgb_img = cv2.merge([r, g, b])
        return rgb_img

    def RGB2BGR(self, img):
        r, g, b = cv2.split(img)
        bgr_img = cv2.merge([b, g, r])
        return bgr_img

    def mosaic_bayer(self, rgb, pattern='BGGR'):
        # analysis pattern
        num = np.zeros(4, dtype=int)
        # the image store in OpenCV using BGR
        temp = list(self.find(pattern, 'R'))
        num[temp] = 0
        temp = list(self.find(pattern, 'G'))
        num[temp] = 1
        temp = list(self.find(pattern, 'B'))
        num[temp] = 2

        mosaic_img = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=rgb.dtype)
        mosaic_img[0::2, 0::2] = rgb[0::2, 0::2, num[0]]
        mosaic_img[0::2, 1::2] = rgb[0::2, 1::2, num[1]]
        mosaic_img[1::2, 0::2] = rgb[1::2, 0::2, num[2]]
        mosaic_img[1::2, 1::2] = rgb[1::2, 1::2, num[3]]
        return mosaic_img

    def WB_Mask(self, img, pattern, fr_now, fb_now):
        wb_mask = np.ones(img.shape)
        if  pattern == 'RGGB':
            wb_mask[0::2, 0::2] = fr_now
            wb_mask[1::2, 1::2] = fb_now
        elif  pattern == 'BGGR':
            wb_mask[1::2, 1::2] = fr_now
            wb_mask[0::2, 0::2] = fb_now
        elif  pattern == 'GRBG':
            wb_mask[0::2, 1::2] = fr_now
            wb_mask[1::2, 0::2] = fb_now
        elif  pattern == 'GBRG':
            wb_mask[1::2, 0::2] = fr_now
            wb_mask[0::2, 1::2] = fb_now
        return wb_mask


    def find(self, str, ch):
        for i, ltr in enumerate(str):
            if ltr == ch:
                yield i

    def Demosaic(self, bayer, pattern='BGGR'):
        results = demosaicing_CFA_Bayer_Malvar2004(bayer, pattern)
        results = np.clip(results, 0, 1)
        return results
    
    def add_PG_noise(self, img, sigma_s='RAN', sigma_c='RAN'):
        min_log = np.log([0.0001])
        if sigma_s == 'RAN':
            sigma_s = min_log + np.random.rand(1) * (np.log([0.16]) - min_log)
            sigma_s = np.exp(sigma_s)
            self.sigma_s = sigma_s
        if sigma_c == 'RAN':
            sigma_c = min_log + np.random.rand(1) * (np.log([0.06]) - min_log)
            sigma_c = np.exp(sigma_c)
            self.sigma_c = sigma_c
        # add noise
        # print('Adding Noise: sigma_s='+str(sigma_s*255)+' sigma_c='+str(sigma_c*255))
        noisy_img = img + \
            np.random.normal(0.0, 1.0, img.shape) * (sigma_s * img) + \
            np.random.normal(0.0, 1.0, img.shape) * sigma_c
        noisy_img = np.clip(noisy_img, 0, 1)
        return noisy_img

    def cbdnet_noise_generate_srgb(self, img):
        # To node that opencv store image in BGR,
        # When apply to color tranfer, BGR should be transfer to RGB
        img_rgb = img
        # -------- INVERSE ISP PROCESS -------------------
        # Step 1 : inverse tone mapping
        icrf_index = random.randint(0, 200)
        img_L = self.ICRF_Map(img_rgb, index=icrf_index)
        # Step 2 : from RGB to XYZ
        img_XYZ = self.RGB2XYZ(img_L)
        # Step 3: from XYZ to Cam
        img_Cam = self.XYZ2CAM(img_XYZ, M_xyz2cam=0)
        # Step 4: Mosaic
        pattern_index = random.randint(0, 3)
        pattern = []
        if pattern_index == 0:
            pattern = 'GRBG'
        elif pattern_index == 1:
            pattern = 'RGGB'
        elif pattern_index == 2:
            pattern = 'GBRG'
        elif pattern_index == 3:
            pattern = 'BGGR'
        self.pattern = pattern
        img_mosaic = self.mosaic_bayer(img_Cam, pattern=pattern)
        # Step 5: inverse White Balance
        min_fc = 0.75
        max_fc = 1
        self.fr_now = random.uniform(min_fc, max_fc)
        self.fb_now = random.uniform(min_fc, max_fc)
        wb_mask = self.WB_Mask(img_mosaic, pattern, self.fr_now, self.fb_now)
        img_mosaic = img_mosaic * wb_mask
        gt_img_mosaic = img_mosaic

        # -------- ADDING POISSON-GAUSSIAN NOISE ON RAW -
        img_mosaic_noise = self.add_PG_noise(img_mosaic)

        # -------- ISP PROCESS --------------------------
        # Step 5 : White Balance
        wb_mask = self.WB_Mask(img_mosaic_noise, pattern, 1/self.fr_now, 1/self.fb_now)
        img_mosaic_noise = img_mosaic_noise * wb_mask
        img_mosaic_noise = np.clip(img_mosaic_noise, 0, 1)
        img_mosaic_gt = gt_img_mosaic * wb_mask
        img_mosaic_gt = np.clip(img_mosaic_gt, 0, 1)
        # Step 4 : Demosaic
        img_demosaic = self.Demosaic(img_mosaic_noise, pattern=self.pattern)
        img_demosaic_gt = self.Demosaic(img_mosaic_gt, pattern=self.pattern)
        # Step 3 : from Cam to XYZ
        img_IXYZ = self.CAM2XYZ(img_demosaic, M_xyz2cam=self.M_xyz2cam)
        img_IXYZ_gt = self.CAM2XYZ(img_demosaic_gt, M_xyz2cam=self.M_xyz2cam)
        # Step 2 : frome XYZ to RGB
        img_IL = self.XYZ2RGB(img_IXYZ)
        img_IL_gt = self.XYZ2RGB(img_IXYZ_gt)
        # Step 1 : tone mapping
        img_Irgb = self.CRF_Map(img_IL, index=icrf_index)
        img_Irgb_gt = self.CRF_Map(img_IL_gt, index=icrf_index)

        return img_Irgb_gt, img_Irgb

    def cbdnet_noise_generate_raw(self, img):
        # To node that opencv store image in BGR,
        # When apply to color tranfer, BGR should be transfer to RGB
        img_rgb = img
        # -------- INVERSE ISP PROCESS -------------------
        # Step 1 : inverse tone mapping
        icrf_index = random.randint(0, 200)
        img_L = self.ICRF_Map(img_rgb, index=icrf_index)
        # Step 2 : from RGB to XYZ
        img_XYZ = self.RGB2XYZ(img_L)
        # Step 3: from XYZ to Cam
        img_Cam = self.XYZ2CAM(img_XYZ, M_xyz2cam=0)
        # Step 4: Mosaic
        pattern_index = random.randint(0, 3)
        pattern = []
        if pattern_index == 0:
            pattern = 'GRBG'
        elif pattern_index == 1:
            pattern = 'RGGB'
        elif pattern_index == 2:
            pattern = 'GBRG'
        elif pattern_index == 3:
            pattern = 'BGGR'
        self.pattern = pattern
        img_mosaic = self.mosaic_bayer(img_Cam, pattern=pattern)
        # Step 5: inverse White Balance
        min_fc = 0.75
        max_fc = 1
        self.fr_now = random.uniform(min_fc, max_fc)
        self.fb_now = random.uniform(min_fc, max_fc)
        wb_mask = self.WB_Mask(img_mosaic, pattern, self.fr_now, self.fb_now)
        img_mosaic = img_mosaic * wb_mask

        # -------- ADDING POISSON-GAUSSIAN NOISE ON RAW -
        img_mosaic_noise = self.add_PG_noise(img_mosaic)
        
        return img_mosaic, img_mosaic_noise


if __name__ == '__main__':
    isp = ISP()
    path = './figs/01_gt.png'
    # To node that opencv store image in BGR,
    # When apply to color tranfer, BGR should be transfer to RGB
    img = cv2.imread(path)
    np.array(img, dtype='uint8')
    img = img.astype('double') / 255.0
    img_rgb = isp.BGR2RGB(img)
    '''
    print('ISP test 1:')
    # -------- INVERSE ISP PROCESS -------------------
    # Step 1 : inverse tone mapping
    img_L = isp.ICRF_Map(img_rgb, index=10)
    # Step 2 : from RGB to XYZ
    img_XYZ = isp.RGB2XYZ(img_L)
    # Step 3: from XYZ to Cam
    xyz2cam = np.array([1.0234, -0.2969, -0.2266, -0.5625, 1.6328, -0.0469, -0.0703, 0.2188, 0.6406])
    img_Cam = isp.XYZ2CAM(img_XYZ, xyz2cam)
    # Step 4: Mosaic
    img_mosaic = isp.mosaic_bayer(img_Cam)

    # -------- ADDING POISSON-GAUSSIAN NOISE ON RAW -
    # Mode1: set sigma_s and sigma_c
    # img_mosaic_noise = isp.add_PG_noise(img_mosaic, sigma_s=0.01, sigma_c=0.02)
    # Mode2: set random sigma_s and sigma_c
    img_mosaic_noise = isp.add_PG_noise(img_mosaic)

    # -------- ISP PROCESS --------------------------
    # Step 4 : Demosaic
    img_demosaic = isp.Demosaic(img_mosaic_noise)
    # Step 3 : from Cam to XYZ
    img_IXYZ = isp.CAM2XYZ(img_demosaic, xyz2cam)
    # Step 2 : frome XYZ to RGB
    img_IL = isp.XYZ2RGB(img_IXYZ)
    # Step 1 : tone mapping
    img_Irgb = isp.CRF_Map(img_IL, index=10)
    '''
  
    '''
    # Observe the images
    show_img = np.concatenate((img,
                               isp.RGB2BGR(img_Irgb),
                               cv2.merge([img_mosaic, img_mosaic, img_mosaic]),
                               isp.RGB2BGR(img_demosaic)
                               ), axis=1)
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Image', show_img)
    cv2.waitKey(0)
    '''

    '''
    print('ISP test 2:')
    gt, noise = isp.cbdnet_noise_generate_srgb(img_rgb)

    # Observe the images
    show_img = np.concatenate((img,
                               isp.RGB2BGR(gt),
                               isp.RGB2BGR(noise)
                               ), axis=1)
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Image', show_img)
    cv2.waitKey(0)
    '''

    print('ISP test 3:')
    gt, noise = isp.cbdnet_noise_generate_raw(img_rgb)
    print(noise_map)
    # Observe the images
    show_img = np.concatenate((img,
                               cv2.merge([noise_map/255, noise_map/255, noise_map/255]),
                               cv2.merge([noise, noise, noise])
                               ), axis=1)
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Image', show_img)
    cv2.waitKey(0)
    
    '''
    img_Ibgr = isp.RGB2BGR(img_Irgb)
    cv2.imwrite('./figs/01_inverse.png', img_Ibgr*255)
    '''
