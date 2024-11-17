# DeepOversampling

This repository provides resources, links, and references for various oversampling techniques, focusing on deep generative models used in machine learning to address class imbalance in datasets. It covers both traditional and modern methods, as discussed in our paper.

## Overview

Imbalanced datasets in machine learning often lead to biased model predictions, as models tend to favor the majority class. Oversampling techniques, including both traditional and deep generative approaches, help mitigate this issue by increasing the representation of minority classes. This repository summarizes key methods and provides links to related academic papers and available code repositories, offering a comprehensive resource for researchers and practitioners in machine learning.

## Table of Contents
* [Traditional Oversampling Techniques](#traditional-oversampling-techniques)
* [Deep Generative Models](#deep-generative-models)
* [Resources and Links](#resources-and-links)

## Traditional Oversampling Techniques

### SMOTE (Synthetic Minority Oversampling Technique):
SMOTE generates new synthetic minority data points by interpolating between several nearest minority class neighbors. For each minority class sample, it calculates the K_NN nearest minority class neighbors (typically K_NN=5) using Euclidean distance. It then creates new synthetic data points along the line segment joining the current minority sample and its selected neighbor. This helps the decision regions associated with the minority class examples to grow larger and less specific.

- **Authors**: N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer
- **Reference**: [SMOTE Paper](https://www.researchgate.net/publication/220543125_SMOTE_Synthetic_Minority_Over-sampling_Technique)
- **Implementations**: [Imbalanced-Learn library](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)

### ADASYN (Adaptive Synthetic Sampling):
ADASYN adaptively generates more synthetic data samples for minority examples that are harder to learn. Unlike SMOTE, which produces the same number of synthetic samples for each minority example, ADASYN decides the number of synthetic examples based on the distribution density of each minority example. This not only reduces bias from imbalanced data distribution but also adaptively shifts the decision boundary toward difficult examples.

- **Authors**: H. He, Y. Bai, E. A. Garcia, and S. Li
- **Reference**: [ADASYN Paper](https://doi.org/10.1109/IJCNN.2008.4633969)
- **Implementations**: [GitHub Implementation](https://github.com/rcamino/deep-generative-models
)**,** [Imbalanced-Learn library](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.ADASYN.html)

### Random Oversampling:
Random oversampling is a simple method that deals with class imbalance by randomly selecting and duplicating minority class examples. While computationally cheap and intuitive, it tends to cause overfitting due to making exact copies of existing data points.
- **Implementations**: [Imbalanced-Learn library](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html)
## Deep Generative Models

### GAN (Generative Adversarial Network):
GANs train a generator network to produce synthetic samples that are indistinguishable from real samples by a discriminator network. The training process is adversarial with the two networks competing against each other. While GANs can capture complex distributions and generate realistic samples, they often face challenges with training stability and mode collapse.

- **Authors**: I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio
- **Reference**: [GAN Paper](https://arxiv.org/abs/1406.2661)
- **Implementations**: [Original Authors' Implementation](https://github.com/goodfeli/adversarial)**,** [GitHub Implementation](https://github.com/rcamino/deep-generative-models)

### CGAN (Conditional GAN):
CGANs extend the classical GAN by conditioning both the generator and discriminator on the class label. This allows for targeted generation of specific minority classes, providing more control over the generation process. However, they still inherit GAN training stability issues.

- **Authors**: M. Mirza and S. Osindero
- **Reference**: [CGAN Paper](https://arxiv.org/abs/1411.1784)
- **Implementations**: [PyTorch Implementation](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

### VAE (Variational Autoencoder):
VAEs learn an explicit latent space representation of the input data using probabilistic encoder and decoder networks. They enable sampling new points from the learned latent distribution but may face issues like posterior collapse and poor reconstruction of complex data.

- **Authors**: D. P. Kingma and M. Welling
- **Reference**: [VAE Paper](https://arxiv.org/abs/1312.6114)
- **Implementations**: [GitHub Implementation](https://github.com/rcamino/deep-generative-models)

### CVAE (Conditional VAE):
CVAEs enable conditioning on class labels for minority class oversampling. The model learns class-dependent distributions by maximizing a lower bound for the conditional distribution. It provides better control over generation but may face higher computational costs.

- **Authors**: K. Sohn, H. Lee, and X. Yan
- **Reference**: [CVAE Paper](https://papers.nips.cc/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html)
### BAGAN (Balancing GAN):
BAGAN is designed specifically for oversampling minority classes in imbalanced image classification. It leverages an autoencoder along with the GAN structure and initializes the decoder as the generator and encoder as part of the discriminator. This initialization enables learning class conditioning in the latent space.

- **Authors**: G. Mariani, F. Scheidegger, R. Istrate, C. Bekas, and C. Malossi
- **Reference**: [BAGAN Paper](https://arxiv.org/abs/1803.09655)

### medGAN:
medGAN combines a GAN with an autoencoder to handle high-dimensional multi-label discrete variables in electronic health records. It uses a pretrained autoencoder to initialize the generator and first layers of the discriminator, enabling effective generation of multi-label discrete data.

- **Authors**: E. Choi, S. Biswal, B. Malin, J. Duke, W. F. Stewart, and J. Sun
- **Reference**: [medGAN Paper](https://arxiv.org/abs/1703.06490)
- **Implementations**: [GitHub Implementation](https://github.com/rcamino/deep-generative-models)

### ARAE (Adversarially Regularized Autoencoder):
ARAE integrates a discrete autoencoder with a latent representation regularized by a GAN. It uses adversarial training to align the latent distribution with a prior, enabling effective modeling of complex discrete distributions.

- **Authors**: J. Zhao, Y. Kim, K. Zhang, A. M. Rush, and Y. LeCun
- **Reference**: [ARAE Paper](https://arxiv.org/abs/1706.04223)

### G-GAN (Gaussian GAN):
G-GAN incorporates prior knowledge about minority class distribution by fitting a Gaussian distribution to minority data. It uses a mixture of Gaussian and uniform distributions for input noise, increasing sample diversity and improving training stability.

- **Authors**: Y. Zhang, Y. Liu, Y. Wang, and J. Yang
- **Reference**: [G-GAN Paper](https://doi.org/10.1016/j.chemolab.2023.104775)

### MoGAN (Minority Oversampling GAN):
MoGAN uses a mixture data distribution for generating minority samples, incorporating majority class distribution information. It employs specialized loss functions and techniques to enhance diversity and reduce bias in generated samples.

- **Authors**: M. Zareapoor, P. Shamsolmoali, and J. Yang
- **Reference**: [MoGAN Paper](https://doi.org/10.1016/j.ymssp.2020.107175)

### 3D-HyperGAMO:
3D-HyperGAMO addresses class imbalance in hyperspectral image classification using a conditional feature mapping unit and patch generator. It retains full spectral information while generating synthetic minority class samples.

- **Authors**: S. K. Roy, J. M. Haut, M. E. Paoletti, S. R. Dubey, and A. Plaza
- **Reference**: [3D-HyperGAMO Paper](https://doi.org/10.1109/TGRS.2021.3052048)

### Improved VAEGAN:
This model enhances the original VAEGAN by adding an extra encoder to improve representation ability. It fuses outputs from two encoders to generate more realistic and diverse minority class data for oversampling.

- **Authors**: Y. Ding, W. Kang, J. Feng, B. Peng, and A. Yang
- **Reference**: [Improved VAEGAN Paper](https://doi.org/10.1109/ACCESS.2023.3302339)

### GANSO:
GANSO combines GANs with vector Markov Random Fields to synthesize realistic instances from limited samples. It incorporates structural information through the vMRF model and uses Graph Fourier Transform for sample generation.

- **Authors**: A. Salazar, L. Vergara, and G. Safont
- **Reference**: [GANSO Paper](https://doi.org/10.1016/j.eswa.2020.113819)
  
**Note**: Not all models have open-source implementations available in standard libraries. In cases where standard library implementations are not available, links direct to the original research papers.

## Resources and Links
- [arXiv](https://arxiv.org/) - Primary source for machine learning and deep learning research papers
- [ResearchGate](https://www.researchgate.net/) - Find additional academic references and publications.
- [IEEE Xplore](https://ieeexplore.ieee.org/) - Digital library for published papers
- [ScienceDirect](https://www.sciencedirect.com/) - A vast repository of scientific and technical research, including papers on machine learning and data science.
- [PyTorch](https://pytorch.org/) - An open-source deep learning framework that provides extensive resources and libraries for implementing machine learning models, including examples of generative models like VAE and GAN.
- [imbalanced-learn](https://imbalanced-learn.org/) - A trusted Python library offering implementations of sampling techniques for handling imbalanced datasets, including popular oversampling methods like SMOTE and ADASYN.







