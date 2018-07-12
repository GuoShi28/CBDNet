# Toward Convolutional Blind Denoising of Real Photographs

## Abstract
Despite their success in Gaussian denoising, deep convolutional neural networks (CNNs) are still very limited on real noisy photographs, and may even perform worse than BM3D. In order to improve the robustness and practicability of deep denoising models, this paper presents a convolutional blind denoising network (CBDNet) by incorporating network architecture, asymmetric learning and noise modeling. Our CBDNet is comprised of a noise estimation subnetwork and a denoising subnetwork. Motivated by the asymmetric sensitivity of BM3D to noise estimation error, the asymmetric learning is presented on the noise 017 estimation subnetwork to suppress more on under-estimation of noise
level. To make the learned model applicable to real photographs, both synthetic images based on signal dependent noise model and real photographs with ground-truth images are incorporated to train our CBDNet. The results on two datasets of real noisy photographs clearly demonstrate the superiority of our CBDNet over the state-of-the-art denoisers in terms of quantitative metrics and perceptual quaility. The data, code and model will be publicly available.

## Network Structure

![Image of Network](figs/CBDNet_v13.png)

## Real Images Denoising Results
### DND dataset
Following the guided of [DND Online submission system](https://noise.visinf.tu-darmstadt.de/).
![Image of DND](figs/DND_results.png)
### Nam dataset
![Image of Nam](figs/Nam_results.png)
## CBDNet Models
* "CBDNet.mat" is the testing model for DND dataset and NC12 dataset for not considering the JPEG compression.
*  "CBDNet_JPEG.mat" is the testing model for Nam dataset and other noisy images with JPEG format.

## Requirements and Dependencies
* Matlab 2015b
* Cuda-8.0 & cuDNN v-5.1
* [MatConvNet](http://www.vlfeat.org/matconvnet/).

## Citation
on going


