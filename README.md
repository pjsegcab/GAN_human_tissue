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
Couple of tests that I ran:
Number of epochs = 20 
Batch size = 100
Resolution of images = 64x64 
Duration = 11 hours

![alt text](https://github.com/pjsegcab/GAN_human_tissue/blob/master/64x64/10.jpg) ![alt text](https://github.com/pjsegcab/GAN_human_tissue/blob/master/64x64/11.jpg) ![alt text](https://github.com/pjsegcab/GAN_human_tissue/blob/master/64x64/14.jpg) ![alt text](https://github.com/pjsegcab/GAN_human_tissue/blob/master/64x64/19.jpg) ![alt text](https://github.com/pjsegcab/GAN_human_tissue/blob/master/64x64/20.jpg) ![alt text](https://github.com/pjsegcab/GAN_human_tissue/blob/master/64x64/21.jpg)

Number of epochs = 10
Batch size = 100
Resolution of images = 128x128 
Duration = 38 hours

![alt text](https://github.com/pjsegcab/GAN_human_tissue/blob/master/128x128/11.jpg) ![alt text](https://github.com/pjsegcab/GAN_human_tissue/blob/master/128x128/18.jpg) ![alt text](https://github.com/pjsegcab/GAN_human_tissue/blob/master/128x128/19.jpg) ![alt text](https://github.com/pjsegcab/GAN_human_tissue/blob/master/128x128/20.jpg) ![alt text](https://github.com/pjsegcab/GAN_human_tissue/blob/master/128x128/22.jpg) ![alt text](https://github.com/pjsegcab/GAN_human_tissue/blob/master/128x128/23.jpg)  

## Limitations
I used a MacBook Pro (i5 2.7 GHz processor, RAM of 8 GB and Graphic card Intel Iris Graphics 6100). The highest decent resolution that I could achieve is 128x128, I tried 512x5112 and the training process did not converge. The hardware is a stopper trying to get higher resolutions for the generated images.




