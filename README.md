# GAN_human_tissue
Implementation of Generative Adversarial Networks (GANs) for the generation of images of human tissues.

## model.py
defines the architecture of the generator, the discriminator and both of them stacked.
## GAN.py
implements the loading and processing of the real dataset, after that loads the models and then performs the training process of the GAN.

## Arguments
Please take a close look at the get_args function, it parses the arguments that you specify: paths, train or generate, number of epochs, etc. 

## Results
I used as a real dataset, 2000 images of skin lessons gotten from ISIC 2017. 
As results I managed to generate 64x64 images, such as:

![alt text](https://github.com/pjsegcab/GAN_human_tissue/blob/master/64x64/10.jpg)


