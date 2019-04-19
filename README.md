# Toward Convolutional Blind Denoising of Real Photographs

### Paper versions
[**CVPR 2019 version**](http://www4.comp.polyu.edu.hk/~cslzhang/paper/CVPR19-CBDNet.pdf)

[**CVPR 2019 supp**](http://www4.comp.polyu.edu.hk/~cslzhang/paper/CVPR19-CBDNet-supp.pdf)

### Add some ISP/inverse ISP operators (Python)
[**Python Code, includimng ISP operators implement, real-world sRGB/RAW noisy image generation and noise map generation**](https://github.com/GuoShi28/CBDNet/blob/master/SomeISP_operator_python/ISP_implement.py)

[**Introduction and examples of the formal code**](https://github.com/GuoShi28/CBDNet/blob/master/SomeISP_operator_python/isp_implement.md)

[**Some simple materials of ISP**](https://github.com/GuoShi28/CBDNet/blob/master/Other_material/some_introduction_material_of_ISP.md)

### Add some matrials 

## 1. Abstract
Despite their success in Gaussian denoising, deep convolutional neural networks (CNNs) are still very limited on real noisy photographs, and may even perform worse than BM3D. In order to improve the robustness and practicability of deep denoising models, this paper presents a convolutional blind denoising network (CBDNet) by incorporating network architecture, asymmetric learning and noise modeling. Our CBDNet is comprised of a noise estimation subnetwork and a denoising subnetwork. Motivated by the asymmetric sensitivity of BM3D to noise estimation error, the asymmetric learning is presented on the noise estimation subnetwork to suppress more on under-estimation of noise
level. To make the learned model applicable to real photographs, both synthetic images based on signal dependent noise model and real photographs with ground-truth images are incorporated to train our CBDNet. The results on two datasets of real noisy photographs clearly demonstrate the superiority of our CBDNet over the state-of-the-art denoisers in terms of quantitative metrics and perceptual quaility. The data, code and model will be publicly available.

## 2. Network Structure

![Image of Network](figs/CBDNet_v13.png)

## 3. Realistic Noise Model
Given a clean image `x`, the realistic noise model can be represented as:

![](http://latex.codecogs.com/gif.latex?\\textbf{y}=f(\\textbf{DM}(\\textbf{L}+n(\\textbf{L}))))

![](http://latex.codecogs.com/gif.latex?n(\\textbf{L})=n_s(\\textbf{L})+n_c)

Where `y` is the noisy image, `f(.)` is the CRF function and the irradiance ![](http://latex.codecogs.com/gif.latex?\\textbf{L}=\\textbf{M}f^{-1}(\\textbf{x})) , `M(.)` represents the function that convert sRGB image to Bayer image and `DM(.)` represents the demosaicing function.

If considering denosing on compressed images, 

![](http://latex.codecogs.com/gif.latex?\\textbf{y}=JPEG(f(\\textbf{DM}(\\textbf{L}+n(\\textbf{L})))))

## 4. Testing
* "Test_Patches.m" is the testing code for small images or image patches. If the tesing image is too large (e.g., 5760*3840), we recommend to use "Test_fullImage.m"
*  "Test_fullImage.m" is the testing code for large images. 
*  "Test_Realistic_Noise_Model.m" is the testing code for the realistic noise mode in our paper. And it's very convinent to utilize [AddNoiseMosai.m](https://github.com/GuoShi28/CBDNet/blob/master/utils/AddNoiseMosai.m) to train your own denoising model for real photographs.

## 5. CBDNet Models
* "CBDNet.mat" is the testing model for DND dataset and NC12 dataset for not considering the JPEG compression.
*  "CBDNet_JPEG.mat" is the testing model for Nam dataset and other noisy images with JPEG format.

## 6. Implement Perceptual Loss Using MatConvnet
The perceptual loss is the MSE loss between the [Perceptual Layer](https://github.com/GuoShi28/CBDNet/tree/master/utils/Perceptual_Layer) outputs of results and labels.
The pretrained vgg model, [fast-rcnn-vgg16-pascal07-dagnn](http://www.vlfeat.org/matconvnet/pretrained/) is needed. 

## 7. Real Images Denoising Results
### 7.1 DND dataset
Following the guided of [DND Online submission system](https://noise.visinf.tu-darmstadt.de/).

![Image of DND](figs/DND_results.png)

### 7.2 Nam dataset

![Image of Nam](figs/Nam_results.png)

## 8. Requirements and Dependencies
* Matlab 2015b
* Cuda-8.0 & cuDNN v-5.1
* [MatConvNet](http://www.vlfeat.org/matconvnet/).

## 9. Citation

```
@article{Guo2019Cbdnet,
  title={Toward convolutional blind denoising of real photographs},
  author={Guo, Shi and Yan, Zifei and Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
  journal={2019 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```

------------------------------------------------------------------------------------------------------------------------------

### Q&A: Why CBDNet can not process some high-noisy photos captured by my own?

A: The main reason is the JPEG compression. For uncompression images even with really high noise under low light condition, CBDNet can remove noise very effectively. Even though we consider JPEG compression on CBDNet, our CBDNet(JPEG) model can only handle jpeg images with normal noise level, e.g., Nam or JPEG compression quality is high.   

I capture some high-noisy images using DLSR camera. Images are stored in both *uncompressed* and *JPEG* format. The denoising results are shown below. 
![](figs/results.png)
