# BUKASA MUYOMBO 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
import numpy as np
import matplotlib.pyplot as plt
import os



import os
print("Current working file:", os.getcwd())

dataset = keras.preprocessing.image_dataset_from_directory("celebA", label_mode = None, image_size= (64,64), batch_size = 40)
#scaling the data in the range [0,1]
dataset = dataset.map(lambda x:x /255.0)
#visualizing the data
for x in dataset:
    plt.axis("off")
    #display the first image in the batch, 
    #x.numpy converts x to a numpy array and is multiplied back to 255, for us to see the image
    plt.imshow((x.numpy() * 255).astype("int32")[0])
    plt.show()
    break



#STEP 2: BUILDING THE DISCRIMINATOR
#the discriminator is expecting an inmput of (64,64) pixels
#this is a binary classifier indicating whether the image is real or not

#we create a sequential model named 'discriminator'
#and we use the keras.Sequential, and it takes a number of layers in the brackets
discriminator = keras.Sequential(
    [
        #input shape is defined, with 3 channels, red, green, and blue
        keras.Input(shape = (64,64,3)),
        #layers.Conv2D(4, kernel_size = 4, strides = 2, padding = "same"),
        #we add a leakyReLu activation layer, and leaky rely introduces a small slope for negative values
        #in this case, it is 0.2 = alpha
        #helps to remove the vanishing gradient problem
        #layers.LeakyReLU(alpha = 0.2),
        #layers.Conv2D(8, kernel_size = 4, strides = 2, padding = "same"),
        #we add a leakyReLu activation layer, and leaky rely introduces a small slope for negative values
        #in this case, it is 0.2 = alpha
        #helps to remove the vanishing gradient problem
        #layers.LeakyReLU(alpha = 0.2),
        #layers.Conv2D(16, kernel_size = 4, strides = 2, padding = "same"),
        #we add a leakyReLu activation layer, and leaky rely introduces a small slope for negative values
        #in this case, it is 0.2 = alpha
        #helps to remove the vanishing gradient problem
        #layers.LeakyReLU(alpha = 0.2),
        #layers.Conv2D(32, kernel_size = 4, strides = 2, padding = "same"),
        #we add a leakyReLu activation layer, and leaky rely introduces a small slope for negative values
        #in this case, it is 0.2 = alpha
        #helps to remove the vanishing gradient problem
        layers.LeakyReLU(alpha = 0.2),
        #we perform convolutional operations on the image with a convoluted layer
        #the kernel size is 4, we have 64 filters, the stride = 2, and padding = same
        layers.Conv2D(64, kernel_size = 4, strides = 2, padding = "same"),
        #we add a leakyReLu activation layer, and leaky rely introduces a small slope for negative values
        #in this case, it is 0.2 = alpha
        #helps to remove the vanishing gradient problem
        layers.LeakyReLU(alpha = 0.2),
        #we also add a flatten layer, it flattens the output into 
        #a 1-dimensional tensor, which can be connected to a fully connected layer 
        layers.Conv2D(128, kernel_size = 4, strides = 2, padding = "same"),
        layers.LeakyReLU(alpha = 0.2),
        layers.Conv2D(256, kernel_size = 4, strides = 2, padding = "same"),
        layers.LeakyReLU(alpha = 0.2),
        layers.Conv2D(512, kernel_size = 4, strides = 2, padding = "same"),
        layers.LeakyReLU(alpha = 0.2),
        
        layers.Flatten(),
        #We then add a dropout layer with a rate of 0.2. 
        #Dropout randomly sets a fraction of the inputs to 0 during training
        #This helps to prevent overfitting
        layers.Dropout(0.2),
        #we then add a dense layer with 1 unit a a sigmoid activation function
        #the output of this layer represents the discriminator's prediction
        #on whether it is real or fake
        layers.Dense(1, activation = "sigmoid"),
        
    
    ],
    name = "discriminator",
)
#we then want a summary of the model's architecture, 
# displaying the layer names, output shapes, and the total number
#of trainable parameters
discriminator.summary()


#STEP 3: THE GENERATOR
#This generator is designed to output images based on the latent space vector as input.
#It is a mirror image of the discriminator, where Conv2D layers are rep;aced
#with conv2dTranspose layers
#these are the layers included here: dense, reshape, transpose, convolutional, and activation

#we set the dimentionality of the latent space (we will investigate with it)
latentDimensions = 128

#we again create a sequential model object named "generator"
#"Using keras.Sequential
# The model will consist of a sequence of layers defined within the brackets
# the layers in the bracket make up the architecture of the model"
generator = keras.Sequential(
    [
        #"We define the input shape of the model, and specify that the input should have a shape of (latentDimensions = 128)
        keras.Input(shape = (latentDimensions,)),

        #"we then add a dense layer with 8*8*128 units
        # This layer serves as the initial decoder
        # and transforms the objects from th latent space to a higher dimensional representation"
        layers.Dense(8*8*128), #8192

        #"we then add a reshape layer to reshape the output of the prevois layer
        # into a 4d tensor with dimensions (8,8,128)
        # this prepares the data for subsequent transpose convolutional layers 
        # that are to come"
        layers.Reshape((8,8,128)),

        #layers.Conv2DTranspose(8, kernel_size = 4, strides = 2, padding = "same"),
        
        #"next we add a leaky realy activation layer with negative slope = 0.2
        # this also introduces non-linearity to the model"
        #layers.LeakyReLU(alpha = 0.2),
        #layers.Conv2DTranspose(16, kernel_size = 4, strides = 2, padding = "same"),
        
        #"next we add a leaky realy activation layer with negative slope = 0.2
        # this also introduces non-linearity to the model"
        #layers.LeakyReLU(alpha = 0.2),
        #layers.Conv2DTranspose(32, kernel_size = 4, strides = 2, padding = "same"),
        
        #"next we add a leaky realy activation layer with negative slope = 0.2
        # this also introduces non-linearity to the model"
        #layers.LeakyReLU(alpha = 0.2),

        #layers.Conv2DTranspose(64, kernel_size = 4, strides = 2, padding = "same"),
        
        #"next we add a leaky realy activation layer with negative slope = 0.2
        # this also introduces non-linearity to the model"
        #layers.LeakyReLU(alpha = 0.2),

        #"we then add a transpose convolutional layer(deconvolution)
        # with 128 filters, a kernel of size 4*4, a stride of 2, 
        # and wih padding as same, this layer upsamples the input"
        layers.Conv2DTranspose(128, kernel_size = 4, strides = 2, padding = "same"),
        
        #"next we add a leaky realy activation layer with negative slope = 0.2
        # this also introduces non-linearity to the model"
        layers.LeakyReLU(alpha = 0.2),

        #"we add another conv2dtranspose layer, but with 256 filers now
        # we will investigate the addition of different number of flters"
        layers.Conv2DTranspose(256, kernel_size = 4, strides = 2, padding = "same"),

        #"we add another leakyReLu layer"
        layers.LeakyReLU(alpha = 0.2),

        #"we add another transpose convolutional layer with 512 layers"
        layers.Conv2DTranspose(512, kernel_size = 4, strides = 2, padding = "same"),

        #"yet another leakyRelu activation function with the same function"
        layers.LeakyReLU(alpha = 0.2),

        #layers.Conv2DTranspose(512, kernel_size = 4, strides = 2, padding = "same"),

        #"yet another leakyRelu activation function with the same function"
        #layers.LeakyReLU(alpha = 0.2),

        #"We then add a final conv2D layer with 3 filters
        # a kernel size of 5*5, padding is still same, and a sigmoid function. this layer produces the output of a generator, representing a generated image"
        layers.Conv2D(3, kernel_size = 5, padding = "same", activation = "tanh"),
        #extract feeatures
    ], name = "generator"
)
#model summary
generator.summary()


#STEP 4: TRAIN THE MODEL
#"We define a training class that includes all the modules, 
# these include the compiling and metrics, as well as training the
# generator and discriminator, as well as updating the metric
# 
# We define a GAN mdel by subclassing the keras model class"

class GAN(keras.Model):
    #the __init__ method initialises the GAN model and class, it taes the 
    # discriminator, generator and latent dimension as arguments and 
    #and uses them as attributes of the GAN
    def __init__ (self, discriminator, generator, latentDimensions):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latentDimensions = latentDimensions

    #The compile method is responsible for setting up the optimizer
    #"And the loss function, and the evaluation of the metrics of the GAN
    # It takes the discriminator, optimizer, gererator optimizer, loss function and input arguments
    # It initializes the loss metrics for the generator and discriminator"
    def compile(self, discOptimizer, genOptimizer, lossFunction):
        super(GAN, self).compile()
        self.discOptimizer = discOptimizer
        self.genOptimizer = genOptimizer
        self.lossFunction = lossFunction
        self.discLossMetric = keras.metrics.Mean(name = "d_loss")
        self.genLossmetric = keras.metrics.Mean(name = "g_loss")

    #"The metrics property returns the list of metrics associated with the model in a tuple
    # in this case, it is the discriminator and generator loss metric"
    @property
    def metrics(self):
        return [self.discLossMetric, self.genLossmetric]
    
    #The train_step method performs a single training step for the GAN moodel
    #and it takes real images as input
    
    def train_step(self, realImages):
        #sample random points in the latent space to make the image generation random
        batch = tf.shape(realImages)[0]
        #returns the value of the input's shape as a tensor
        randomLatentVectors = tf.random.normal(shape = (batch, self.latentDimensions))

        #we will now decode them to "fake" images
        #the generated images use the latent space representation vector
        newImags = self.generator(randomLatentVectors)

        #and now we combine them with real images in order to fool the decoder
        combinedImages = tf.concat([newImags, realImages], axis = 0)

        #and then we assemble the labels that will differentiate real and the fake generated images
        labels = tf.concat([tf.zeros((batch, 1)), tf.ones((batch,1))], axis = 0)

        #and then we can add random noise to the labels, this is important as it is label smoothing,
        #and it prevents overfitting

        #and we do that by generating uniform random numbers
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        #The next step is to train the discriminator
        with tf.GradientTape() as gt:
            #stakes in the combined images
            predictions = self.discriminator(combinedImages)
            d_loss = self.lossFunction(labels, predictions)
        dGradient = gt.gradient(d_loss, self.discriminator.trainable_weights)
        self.discOptimizer.apply_gradients(zip(dGradient, self.discriminator.trainable_weights))

        #next we sample random points in the latent space again
        randomLatentVectors = tf.random.normal(shape = (batch, self.latentDimensions))

        #We assembe labels that say all real images
        #
        wrongLabels = tf.zeros((batch, 1)) #these are the labels that are wromg 
        #and belong to the wrong class

        #The next step is to train the generator, and we need to 
        #"make sure that we do not update the weights of the discriminator
        # becuase we want them to train individually
        
        with tf.GradientTape() as gt:
            genImgs = self.generator(randomLatentVectors)
            predictions = self.discriminator(genImgs)
            g_loss = self.lossFunction(wrongLabels, predictions)
        
        genGradient = gt.gradient(g_loss, self.generator.trainable_weights)
        self.genOptimizer.apply_gradients(zip(genGradient, self.generator.trainable_weights))

        #WE then update the metrics
        self.discLossMetric.update_state(d_loss)
        self.genLossmetric.update_state(g_loss)

        return {
            "d_loss": self.discLossMetric.result(),
            "g_loss": self.genLossmetric.result(),
        }

#We create a GANMonitor class which generates a specific number of images
#at the end of each training epoch, and then sames them as png files

class GANMonitor(keras.callbacks.Callback):
    #the constructor initializes the num-img and latentDimensions attributes with the default value
    #but we can also provide new values as the arguments
    def __init__(self, numOfImages = 10, latentDimensions = 128):
        self.numOfImages = numOfImages
        self.latentDimensions = latentDimensions

    #"This method is called at the end of each traininng epoch
    # It is responsible for generating random latent vectors from a normal distribution
    # these latent vectors have a shape of (number of images, latent dimention)
    # the latent vectors are passed through the generator model (self.model.generator)
    # resulting in generated images, with pixels that are scaled up by multiplying by 255.
    # Then the numpy method is used to convert the generated images to numpy arrays"
    def on_epochs_end(self, epoch):
        randomLatentVectors = tf.random.normal(shape = (self.numOfImages, self.latentDimensions))
        newImags = self.model.generator(randomLatentVectors)
        newImags *= 255
        newImags = newImags.np()

        #"Next we iterate over the generated images, and convert them 
        # to a png.
        # Using keras.processing.image.array_to_img() method
        # Then it saves the image using the save() method with a filename format
        # the format uses an epoch number and the index of the image"

        for i in range(self.numOfImages):
            img = keras.preprocessing.image.array_to_img(newImags[i])
            img.save("imgage_%03d_%d.png" % (epoch,i))

#The GAN model trains on the specified data, while the GANMonitor
# monitors the training

#this value of epoch = 1 is assign to specify the number of trainig epochs
epochs = 100
#we create an instance of the GAN model by passing the  discriminator, generator,
#latent dimension as arguments
gan = GAN(discriminator=discriminator, generator= generator, latentDimensions=latentDimensions)

#next, we compile the gan model, whcih specifies the
#"discriminator's oprimizer -> discOptimizer
# the generator's optimizer -> genOptimizer
# the loss function -> lossFunction to be used in training
# 
# with the learning rate of 0.0001 used for both the discriminator and generator,
# snd the binarycrossentropy loss function is used"

# we will investigate with the learning rate, and optimizer
gan.compile(
    discOptimizer=keras.optimizers.Adam(learning_rate= 0.0002),
    genOptimizer=keras.optimizers.Adam(learning_rate= 0.0001),
    lossFunction=keras.losses.BinaryCrossentropy()
)

#we starts the process for training the gan model
#we train it based on the dataset for the specified number of epochs
#also, we use the ganmonitor callback during training
#this generates and saves 10 images at the end
#and provides a visual for us to monitor
gan.fit(dataset, epochs=epochs, callbacks=[GANMonitor(numOfImages=10, latentDimensions=latentDimensions)])