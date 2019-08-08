# -*- coding: utf-8 -*-
# Power by Zongsheng Yue 2019-06-11 22:29:36
'''
ISP class
============================================================================================================= 
include main operators of in-camera processing pipeline (ISP) and some corresponding inverse operators.

Reference: https://github.com/GuoShi28/CBDNet

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
import pyximport; pyximport.install()
from tone_mapping_cython import CRF_Map_Cython, ICRF_Map_Cython
import time
import matplotlib.pyplot as plt

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
                            ,[0.6596,-0.2079,-0.0562,-0.4782,1.3016,0.1933,-0.097,0.1581,0.5181]])

    def ICRF_Map(self, img, index=0):
        invI_temp = self.I_inv[index, :]
        invB_temp = self.B_inv[index, :]
        out = ICRF_Map_Cython(img.astype(np.double), invI_temp.astype(np.double), invB_temp.astype(np.double))
        return out

    def CRF_Map(self, img, index=0):
        I_temp = self.I[index, :]  # shape: (1024, 1)
        B_temp = self.B[index, :]  # shape: (1024, 1)
        out = CRF_Map_Cython(img.astype(np.double), I_temp.astype(np.double), B_temp.astype(np.double))
        return out

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
        return img[:, :, ::-1]

    def RGB2BGR(self, img):
        return img[:, :, ::-1]

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
        sigma_total = np.sqrt(sigma_s * img + sigma_c)
        noisy_img = img +  \
            sigma_total * np.random.randn(img.shape[0], img.shape[1])
        noisy_img = np.clip(noisy_img, 0, 1)
        return noisy_img

    def cbdnet_noise_generate_srgb(self, img):
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
    isp = ISP('./')
    path = './01_gt.png'
    # To node that opencv store image in BGR,
    img = cv2.imread(path)[:,:,::-1]
    np.array(img, dtype='uint8')
    img = img.astype('double') / 255.0

    print('ISP test step by step:')
    # -------- INVERSE ISP PROCESS -------------------
    # Step 1 : inverse tone mapping
    img_L = isp.ICRF_Map(img, index=10)
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
    fig = plt.figure(0)
    fig.canvas.set_window_title('Step test')
    plt.imshow(img_Irgb)

    print('ISP test on sRGB space:')
    im_gt, im_noisy = isp.cbdnet_noise_generate_srgb(img)
    fig = plt.figure(1)
    fig.canvas.set_window_title('sRGB')
    plt.imshow(np.concatenate((im_gt, im_noisy), axis=1))
    plt.show()


