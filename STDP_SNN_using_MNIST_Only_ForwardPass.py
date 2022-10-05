#!/usr/bin/env python
# coding: utf-8

# ## STDP implementation using MNIST Data

# #### Importing necessary Libs

# In[58]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)


# #### Loading MNIST Dataset

# In[60]:


get_ipython().system('pip3 install python-mnist')
# from mnist.loader import MNIST
# import os
# loader = MNIST(os.path.join(os.path.dirname(os.getcwd())))

(train_images, train_labels), (
    test_images,
    test_labels,
) = tf.keras.datasets.mnist.load_data()

# loader returns a list of 768-element lists of pixel values in [0,255]
# and a corresponding array of single-byte labels in [0-9]


## reducing the number of images to be processed by selecting a few from each label class
_1, train_images, _2, train_labels = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)
_3, test_images, _4, test_labels = train_test_split(test_images, test_labels, test_size=0.1, random_state=42)


n_train = len(train_labels)
n_test = len(test_labels)

print('n_train: ',n_train,', n_test: ',n_test)

# TrainIm, TrainL = loader.load_training()
TrainIm = np.array(train_images) # convert to ndarray
TrainL = np.array(train_labels)
TrainIm = TrainIm / TrainIm.max() # scale to [0, 1] interval

# TestIm, TestL = loader.load_testing()
TestIm = np.array(test_images) # convert to ndarray
TestL = np.array(test_labels)
TestIm = TestIm / TestIm.max() # scale to [0, 1] interval

# Randomly select train and test samples
trainInd = np.random.choice(len(TrainIm), n_train, replace=False)
TrainIm = TrainIm[trainInd]
TrainLabels = TrainL[trainInd]

testInd = np.random.choice(len(TestIm), n_test, replace=False)
TestIm = TestIm[testInd]
TestLabels = TestL[testInd]


# In[56]:


set(train_labels)


# In[42]:


train_images.shape


# #### Create the NN model 

# In[6]:


# input
inp = tf.keras.Input(shape=(28, 28, 1))

# convolutional layers
conv0 = tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=3,
    activation=tf.nn.relu,
)(inp)

conv1 = tf.keras.layers.Conv2D(
    filters=64,
    kernel_size=3,
    strides=2,
    activation=tf.nn.relu,
)(conv0)

# fully connected layer
flatten = tf.keras.layers.Flatten()(conv1)
dense = tf.keras.layers.Dense(units=10)(flatten)

model = tf.keras.Model(inputs=inp, outputs=dense)


# In[7]:


inp.shape


# #### Convert Images to Spiking Matrix

# In[12]:


## Poisson Spike Mat Generator
def gen_Poisson_spikes(img_array, fr, bins, spike_train):
    img_array = img_array.flatten()
    dt = 0.001
    for pixel in range(img_array.size):
        if img_array[pixel] != 0:
            fr2 = fr * img_array[pixel]
            poisson_output = np.random.rand(1, bins) < fr2 * dt
            spike_train[pixel] = poisson_output.astype(int)

    return spike_train


# In[48]:


# Learning rule parameters
Imin = -0.05 # Minimum current for learning
Imax = 0.05 # Maximum current for learning
lr0 = 1e-4 # Learning rate in hidden layer
w_scale0 = 1e-3 # Weight scale in hidden layer
lr1 = 1e-5 # Learning rate at output layer
lrP = 1
lrN = 1
w_scale1 = 1e-3 # Weight scale at output layer
FPF = 0 # inhibits punishing target neuron (only use if training a specific output spike pattern)

# Neuron parameters
t_syn = 25
t_syn1 = 25 # Synaptic time-constant at output layer

t_m = 25 # Neuron time constant in hidden layer
t_mH = 10 # Neuron time constant in output layer
t_mU = 150 # Neuron time constant for error accumulation
t_mE = 150 # Error neuron time constant

R = t_m/10 # Membrane resistance in hidden layer
RH = t_mH/50 # Membrane resistance in output layer
RE = t_mE/150 # Membrane resistance in error neurons
RU = t_mU/150 # Membrane resistance of error compartment

Vth = 0.005 # Hidden neuron threshold
VthO = 0.005 # Output neuron threshold
VthE = 0.01 # Error neuron threshold

V_rest = 0 # Resting membrane potential
t_refr = 4 # Duration of refractory period

# Simulation parameters
tSim = 0.15 # Duration of simulation (seconds)
MaxF = 250 # maximum frequency of the input spikes
maxFL = 500 # maximum frequency of the target label spikes
dt = 1 # time resolution
dt_conv = 1e-3 # Data is sampled in ms
nBins = int(np.floor(tSim/dt_conv)) #total no. of time steps

# Network architecture parameters
n_h1 = 200 # no. of hidden neurons
dim = 28 # dim by dim is the dimension of the input images
n_in = dim*dim # no. of input neurons
n_out = 10 # no. of output neurons

# Generate forward pass weights
w_in = np.random.normal(0, w_scale0, (n_h1, n_in)) # input->hidden weights
w_out = np.random.normal(0, w_scale1, (n_out, n_h1)) # hidden->output weights

# Generate random feedback weights
## SWITCH OFF Feeback 
# w_err_h1p = np.random.uniform(-1, 1, (n_h1, n_out)) # false-pos-error->hidden weights
# w_err_h1n = np.random.uniform(-1, 1, (n_h1, n_out)) # false-neg-error->hidden weights

# MNIST Parameter
maxE = 5  #no. of epochs
train_acc = np.zeros((maxE))
test_acc = np.zeros_like(train_acc)


# In[51]:


def make_spike_trains(freqs, n_steps):
    ''' Create an array of Poisson spike trains
        Parameters:
            freqs: Array of mean spiking frequencies.
            n_steps: Number of time steps
    '''
    r = np.random.rand(len(freqs), n_steps)
    spike_trains = np.where(r <= np.reshape(freqs, (len(freqs),1)), 1, 0)
    return spike_trains

def MNIST_to_Spikes(maxF, im, t_sim, dt):
    ''' Generate spike train array from MNIST image.
        Parameters:
            maxF: max frequency, corresponding to 1.0 pixel value
            FR: MNIST image (784,)
            t_sim: duration of sample presentation (seconds)
            dt: simulation time step (seconds)
    '''
    n_steps = np.int(t_sim / dt) #  sample presentation duration in sim steps
    freqs = im.flatten() * maxF * dt # scale [0,1] pixel values to [0,maxF] and flatten
    return make_spike_trains(freqs, n_steps)


# In[63]:


for e in range(maxE): # for each epoch
    correct_predictions = 0
    test_predictions = 0
    for u in range(n_train): # for each training pattern
        # Generate poisson data and labels
        spikeMat = MNIST_to_Spikes(MaxF, TrainIm[u][:], tSim, dt_conv)
        fr_label = np.zeros(n_out)
        fr_label[TrainLabels[u]] = maxFL # target output spiking frequencies
        s_label = make_spike_trains(fr_label * dt_conv, nBins) # target spikes

        # Initialize hidden layer variables
        I1 = np.zeros(n_h1)
        V1 = np.zeros(n_h1)
        U1 = np.zeros(n_h1)

        # Initialize output layer variables
        I2 = np.zeros(n_out)
        V2 = np.zeros(n_out)
        U2 = np.zeros(n_out)

        # Initialize error neuron variables
        Verr1 = np.zeros(n_out)
        Verr2 = np.zeros(n_out)

        # Initialize firing time variables
        ts1 = np.full(n_h1, -t_refr)
        ts2 = np.full(n_out, -t_refr)
        tsE1 = np.full(n_out, -t_refr)
        tsE2 = np.full(n_out, -t_refr)

        SE1T = np.zeros((10, nBins)) # to record error neuron spiking
        SE2T = np.zeros((10, nBins))
        train_counter = np.zeros((n_out))
        for t in range(nBins):
            # Forward pass
           
            # Find input neurons that spike
            fired_in = np.nonzero(spikeMat[:, t])
            # print (len(fired_in))
            # Update synaptic current into hidden layer
            I1 = I1 + (dt/t_syn) * (w_in.dot(spikeMat[:, t]) - I1)

            # Update hidden layer membrane potentials
            V1 = V1 + (dt/t_m) * ((V_rest - V1) + I1 * R)
            V1[V1 < -Vth/10] = -Vth/10 # Limit negative potential

            # If neuron in refractory period, prevent changes to membrane potential
            refr1 = (t*dt - ts1 <= t_refr)
            V1[refr1] = 0

            fired = np.nonzero(V1 >= Vth) # Hidden neurons that spiked
            V1[fired] = 0 # Reset their membrane potential to zero
            ts1[fired] = t # Update their most recent spike times

            ST1 = np.zeros(n_h1) # Hidden layer spiking activity
            ST1[fired] = 1 # Set neurons that spiked to 1

            # Repeat the process for the output layer
            I2 = I2 + (dt/t_syn1)*(w_out.dot(ST1) - I2)

            V2 = V2 + (dt/t_mH)*((V_rest - V2) + I2*(RH))
            V2[V2 < -VthO/10] = -VthO/10

            refr2 = (t*dt - ts2 <= t_refr)
            V2[refr2] = 0
            fired2 = np.nonzero(V2 >= VthO)

            V2[fired2] = 0
            ts2[fired2] = t

            s2 = np.zeros((n_out))
            s2[fired2] = 1
            train_counter = train_counter + s2
  
            for output_neuron in range(n_out):
                if (Imin <= I2[output_neuron] and I2[output_neuron] <= Imax):
                    w_out[output_neuron, fired] -= lr1 * U2[output_neuron]

    # TODO: Check train and test accuracy here.
    # If the output neuron with highest firing rate matches the target
    # neuron, and that rate is > 0, then the sample was classified correctly
        tn = TrainLabels[u]
        wn = np.argmax(train_counter)
        # print (train_counter, wn, tn)
        if tn==wn:
            correct_predictions += 1

    print (f'Accuracy in epoch {e} is {(correct_predictions/n_train)} {correct_predictions}')
    train_acc[e] = (correct_predictions/n_train)*100
    ## Perform Testing

    for u in range(n_test):  # for each training pattern
        # Generate poisson data and labels
        spikeMat = MNIST_to_Spikes(MaxF, TestIm[u], tSim, dt_conv)
        fr_label = np.zeros(n_out)
        fr_label[TestLabels[u]] = maxFL  # target output spiking frequencies
        s_label = make_spike_trains(fr_label * dt_conv, nBins)  # target spikes

        # Initialize hidden layer variables
        I1 = np.zeros(n_h1)
        V1 = np.zeros(n_h1)
        U1 = np.zeros(n_h1)

        # Initialize output layer variables
        I2 = np.zeros(n_out)
        V2 = np.zeros(n_out)
        U2 = np.zeros(n_out)

        # Initialize error neuron variables
        Verr1 = np.zeros(n_out)
        Verr2 = np.zeros(n_out)

        # Initialize firing time variables
        ts1 = np.full(n_h1, -t_refr)
        ts2 = np.full(n_out, -t_refr)
        tsE1 = np.full(n_out, -t_refr)
        tsE2 = np.full(n_out, -t_refr)

        SE1T = np.zeros((10, nBins))  # to record error neuron spiking
        SE2T = np.zeros((10, nBins))
        test_counter = np.zeros((n_out))

        for t in range(nBins):
            # Forward pass

            # Find input neurons that spike
            fired_in = np.nonzero(spikeMat[:, t])
            # print (len(fired_in))
            # Update synaptic current into hidden layer
            I1 = I1 + (dt / t_syn) * (w_in.dot(spikeMat[:, t]) - I1)

            # Update hidden layer membrane potentials
            V1 = V1 + (dt / t_m) * ((V_rest - V1) + I1 * R)
            V1[V1 < -Vth / 10] = -Vth / 10  # Limit negative potential

            # If neuron in refractory period, prevent changes to membrane potential
            refr1 = (t * dt - ts1 <= t_refr)
            V1[refr1] = 0

            fired = np.nonzero(V1 >= Vth)  # Hidden neurons that spiked
            V1[fired] = 0  # Reset their membrane potential to zero
            ts1[fired] = t  # Update their most recent spike times

            ST1 = np.zeros(n_h1)  # Hidden layer spiking activity
            ST1[fired] = 1  # Set neurons that spiked to 1

            # Repeat the process for the output layer
            I2 = I2 + (dt / t_syn1) * (w_out.dot(ST1) - I2)

            V2 = V2 + (dt / t_mH) * ((V_rest - V2) + I2 * (RH))
            V2[V2 < -VthO / 10] = -VthO / 10

            refr2 = (t * dt - ts2 <= t_refr)
            V2[refr2] = 0
            fired2 = np.nonzero(V2 >= VthO)

            V2[fired2] = 0
            ts2[fired2] = t

            s2 = np.zeros((n_out))
            s2[fired2] = 1

            test_counter = test_counter + s2

  
        tn = TestLabels[u]
        wn = np.argmax(test_counter)
        if tn==wn:
            test_predictions += 1

    print (f' Test Accuracy in epoch {e} is {(test_predictions/n_test)} {test_predictions}')
    test_acc[e] = 100*(test_predictions/n_test)

# Generating plots
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
plt.plot(train_acc, color='blue', label="Train Accuracy")
plt.plot(test_acc, color='red', label='Test Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Model Performance")
plt.title("Froward Pass Performance on MNIST Classification")
plt.legend()
plt.show()


# In[ ]:




