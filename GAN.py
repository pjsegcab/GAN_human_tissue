
import sys
import argparse
import cv2
import numpy as np
import glob
import os
import model
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from keras.optimizers import Adam
from keras.optimizers import SGD, RMSprop
import os, struct
from array import array as pyarray

print(sys.executable)

def load_image(path):
    ''' Resize image to 64x64 and shuffle axis to create 3 arrays (RGB) '''
    img = cv2.imread(path, 1)
    img = np.float32(cv2.resize(img, (64, 64))) / 127.5 - 1
    img = np.rollaxis(img, 2, 0)

    return img


def noise_image():
    ''' Create noisy data that will be converted to an image
        Note size = (total number, number in sublist, length of subsublist )
    '''
    zmb = np.random.uniform(-1, 1, 100)
    #zmb = np.random.uniform(0, 1, 100)
    return zmb


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]


def train(path, batch_size, EPOCHS):

    #reproducibility
    np.random.seed(123)

    fig = plt.figure()

    # Get image paths
    print("Loading paths..")
    paths = glob.glob(os.path.join(path, "*.jpg"))
    
    print("Got paths..")

    # Load images
    IMAGES = np.array([load_image(p) for p in paths])
    np.random.shuffle(IMAGES)

    # Organize the images into batches
    BATCHES = [b for b in chunks(IMAGES, batch_size)]

    # Instantiate the discriminator, the generator and the gen_containing_disc
    discriminator = model.discriminator_model()
    generator = model.generator_model()
    discriminator_on_generator = model.generator_containing_discriminator(generator, discriminator)

    # Define the optimizers
    adam_gen = Adam(lr = 0.00002, beta_1 = 0.0005, beta_2 = 0.999, epsilon = 1e-08)
    adam_dis = Adam(lr = 0.00002, beta_1 = 0.0005, beta_2 = 0.999, epsilon = 1e-08)
    
    # Compile the models
    generator.compile(loss = 'binary_crossentropy', optimizer = adam_gen)
    discriminator_on_generator.compile(loss = 'binary_crossentropy', optimizer = adam_gen)
    discriminator.trainable = True
    discriminator.compile(loss = 'binary_crossentropy', optimizer = adam_dis)

    print("Number of batches", len(BATCHES))
    print("Batch size is", batch_size)

    # Define intermodel margin
    inter_model_margin = 0.1

    # Training core loop
    for epoch in range(EPOCHS):
        print()
        print("Epoch", epoch)
        print()

        # Load weights on first try (i.e. if process failed previously and we are attempting 
        # to recapture lost data)
        if epoch == 0:
            if os.path.exists('generator_weights') and os.path.exists('discriminator_weights'):
                print("Loading saved weights..")
                generator.load_weights('generator_weights')
                discriminator.load_weights('discriminator_weights')
                print("Finished loading")
            else:
                pass

        # Looping inside of a batch                    
        for index, image_batch in enumerate(BATCHES):
            print("Epoch", epoch, "Batch", index)

            # Create a batch of noise vectors 
            Noise_batch = np.array([noise_image() for n in range(len(image_batch))])

            # Generate images out of the noise batch
            generated_images = generator.predict(Noise_batch)

            # Saving all the generated images 
            for i, img in enumerate(generated_images):
                rolled = np.rollaxis(img, 0, 3)
                if os.path.exists('./results'):
                	cv2.imwrite('./results/' + 'Epoch_' + str(epoch) + '_' + str(i) + ".jpg", np.uint8(255 * 0.5 * (rolled + 1.0)))
                else:
                	os.mkdir('./results')
                	cv2.imwrite('./results/' + 'Epoch_' + str(epoch) + '_' + str(i) + ".jpg", np.uint8(255 * 0.5 * (rolled + 1.0)))

            # Defining input of the discriminator
            X_disc = np.concatenate((image_batch, generated_images))
            
            # Defining labels for the discriminator		
            Y_disc = [1] * len(image_batch) + [0] * len(image_batch) # labels

            print("Training first discriminator...")
            
            # Discriminator training
            d_loss = discriminator.train_on_batch(X_disc, Y_disc)

            # Defining input of the generator
            X_gen = Noise_batch

            # Defining labels for the generator
            Y_gen = [1] * len(image_batch)

            print("Training first generator...")
            g_loss = discriminator_on_generator.train_on_batch(X_gen, Y_gen)

            print("Initial batch losses : ", "Generator loss", g_loss, "Discriminator loss", d_loss, "Total:", g_loss + d_loss)

            # Convergence conditions
            if g_loss < d_loss and abs(d_loss - g_loss) > inter_model_margin:
                while abs(d_loss - g_loss) > inter_model_margin:
                    print(abs(d_loss - g_loss))
                    print("Updating discriminator..")
                    d_loss = discriminator.train_on_batch(X_disc, Y_disc)
                    print("Generator loss", g_loss, "Discriminator loss", d_loss)
                    if d_loss < g_loss:
                        break
            elif d_loss < g_loss and abs(d_loss - g_loss) > inter_model_margin:
                while abs(d_loss - g_loss) > inter_model_margin:
                    print(abs(d_loss - g_loss))
                    print("Updating generator..")
                    g_loss = discriminator_on_generator.train_on_batch(X_gen, Y_gen)
                    print("Generator loss", g_loss, "Discriminator loss", d_loss)
                    if g_loss < d_loss:
                        break
            else:
                pass

            print("Final batch losses (after updates) : ", "Generator loss", g_loss, "Discriminator loss", d_loss, "Total:", g_loss + d_loss)
            print()

            # Saving weights
            print('Saving weights..')
            generator.save_weights('generator_weights', True)
            discriminator.save_weights('discriminator_weights', True)

        plt.clf()

        # Show a combined plot generated images for visual assessment
        for i, img in enumerate(generated_images[:5]):
            i = i + 1
            plt.subplot(3, 3, i)
            rolled = np.rollaxis(img, 0, 3)
            plt.imshow(rolled)
            plt.axis('off')
        fig.canvas.draw()
        plt.savefig('Epoch_' + str(epoch) + '.png')


def generate(img_num):
    '''
        Generate new images based on trained model.
    '''
    generator = model.generator_model()
    adam = Adam(lr = 0.00002, beta_1 = 0.0005, beta_2 = 0.999, epsilon = 1e-08)
    generator.compile(loss = 'binary_crossentropy', optimizer = adam)
    generator.load_weights('generator_weights')

    noise = np.array([noise_image() for n in range(img_num)])

    print('Generating images..')
    generated_images = [np.rollaxis(img, 0, 3) for img in generator.predict(noise)]
    for index, img in enumerate(generated_images):
        cv2.imwrite("{}.jpg".format(index), np.uint8(255 * 0.5 * (img + 1.0)))
    print(np.shape(generated_images[0]))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type = str, default = "../Dataset/ISIC_2017/")
    parser.add_argument("--TYPE", type = str, default = "generate")
    parser.add_argument("--batch_size", type = int, default = 100)
    parser.add_argument("--epochs", type = int, default = 10)
    parser.add_argument("--img_num", type = int, default = 25)

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_args()
    
    if args.TYPE == 'train':
        train(path = args.path, batch_size = args.batch_size, EPOCHS = args.epochs)
    elif args.TYPE == 'generate':
        generate(img_num = args.img_num)










