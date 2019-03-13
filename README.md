# GAN_human_tissue
Implementation of Generative Adversarial Networks (GANs) for the generation of images of human tissue.

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

In order to validate the output quality, I carried out an analysis (tSNE and PCA) to compare the statistical variability of the FAKE VS. REAL images. Highlighted with blue arrows are the fake images.  
Notice that they are distributed along the graph, they do not form a cluster theirselves. This means the statistical variability is similar for both fake and real images and the generated ones can be added to the real dataset as if they were real ones for further studies.  

_tSNE for 128x128 images resolution:_  
![alt text](https://github.com/pjsegcab/GAN_human_tissue/blob/master/Validation/tSNE_128px.png)  

_PCA for 128x128 images resolution:_  
![alt text](https://github.com/pjsegcab/GAN_human_tissue/blob/master/Validation/PCA_128px.png)  


## Limitations
HARDWARE. I used a MacBook Pro (i5 2.7 GHz processor, RAM of 8 GB and Graphic card Intel Iris Graphics 6100). The highest decent resolution that I could achieve is 128x128, I tried 512x512 and the training process did not converge. The hardware is a stopper trying to get higher resolutions for the generated images.




