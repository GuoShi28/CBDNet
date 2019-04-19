## 1. Introduction
Here we complete the noise generation process of CBDNet.

The code is released for users to better reproduce real-world denoising works better using python.

Hope this helps. If there are some questions or mistakes in my code, please feel free to contact me.

## 2. Major process of CBDNet+ noise generation

### noise generation on sRGB Space

**INPUT:** clean sRGB image ``x`` 

**Step 1:** inverse [tone mapping](https://en.wikipedia.org/wiki/Tone_mapping). 

We uniformly sampled 201 CRFs and using these curve to implement tone mapping and inverse tone mapping.

Please refer to the CBDNet paper and released code for details.

![](http://latex.codecogs.com/gif.latex?\\textbf{x}_L=\text{ICRFMap}(\\textbf{x}))

**Step 2:** inverse Color Correction (Not included in CBDNet paper)

In image processing pipeling, the color spave transfer from cam to xyz and finally tranfer to sRGB.

These process also affect the distribution of real-world noise.

For better understand the process of Color Correction, we highly recommand readers to also refer to a PPT from Michael S. Brown, [Understanding the In-Camera Image Processing Pipeline for Computer Vision](https://www.eecs.yorku.ca/~mbrown/CVPR2016_Brown.html).

![](http://latex.codecogs.com/gif.latex?\\textbf{x}_{\textbf{xyz}}=\text{RGB2XYZ}(\\textbf{x}_L))

![](http://latex.codecogs.com/gif.latex?\\textbf{x}_{\textbf{cam}}=\text{XYZ2CAM}(\\textbf{x}_{\textbf{xyz}}))

**Step 3:** Mosaic

In this step, the 3 channel image is transfer to bayer image.

![](http://latex.codecogs.com/gif.latex?\\textbf{x}_{\textbf{mosaic}}=\text{M}(\\textbf{x}_{\textbf{cam}}))

**Step 4:** Inverse white balance (Not included in CBDNet paper)

![](http://latex.codecogs.com/gif.latex?\\textbf{x}_{\textbf{mosaic}}=\textbf{x}_{\textbf{mosaic}}*\textbf{wb}_{\textbf{mask}}^{-1})

**Step 5:** Adding Poission-gaussion noise

Different from original CBDNet paper, we also refer to [Unprocessing Images for Learned Raw Denoising](https://arxiv.org/pdf/1811.11127.pdf).

There are two major difference between this version and original CBDNet paper.

1. The noise levels are uniformly distribut on log space. This process can let the network focus more on normal noise level. 

2. The noise map definition change to a more strict form. 

For sRGB denoising, the noise map is defined as 6-channel feature which represent the various of R,G,B and covariance of RG, GB, RB.

For RAW denoising, the noise map is also defined as the various of the noise image, which is almost the same with original CBDNet paper.

**Other Steps** ISP process to transfer raw nosie image back to sRGB space

The steps are the isp process and are the inverse steps of Step1 to Step4.

For details, please refer to the code directly.

### noise generation on RAW Space

Very similar to ```noise generation on sRGB Space```.

For details, please refer to the code directly.

## 3. Testing Demo

### real-world sRGB noise image generation

```
isp = ISP()
path = './figs/01_gt.png'
# To node that opencv store image in BGR,
# When apply to color tranfer, BGR should be transfer to RGB
img = cv2.imread(path)
np.array(img, dtype='uint8')
img = img.astype('double') / 255.0
img_rgb = isp.BGR2RGB(img)

gt, noise = isp.cbdnet_noise_generate_srgb(img_rgb)
noise_map = isp.cal_noise_map_srgb(noise*255, gt*255, patch_size=8)/255
```

### real-world raw noise image generation

```
isp = ISP()
path = './figs/01_gt.png'
# To node that opencv store image in BGR,
# When apply to color tranfer, BGR should be transfer to RGB
img = cv2.imread(path)
np.array(img, dtype='uint8')
img = img.astype('double') / 255.0
img_rgb = isp.BGR2RGB(img)

gt, noise = isp.cbdnet_noise_generate_raw(img_rgb)
noise_map = isp.cal_noise_map_raw(noise*255, gt*255, patch_size=8)/255
```
