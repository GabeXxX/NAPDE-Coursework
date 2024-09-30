# Bayesian NN for the cardiovascular model based on
# Solving Bayesian Inverse Problems via Variational Autoencoders (2022?) Tan Bui-Thanh
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Suppress initial tensorflow message
import numpy as np
import csv
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
from keras import layers
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import joblib

# Scale x data to [-1, 1]
def scale(x,dimHt):
    # Fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))

    # Remove test dataset and reshape train from 3D to 2D array
    x2Fit = x[:9*dimHt,:]
    scaler = scaler.fit(x2Fit)
    # Transform train
    xScaled = scaler.transform(x)
    
    return xScaled, scaler

def create_uqvae(nInputs : int,nOutputs : int):
    neurons = nOutputs*(nOutputs+1)//2+nOutputs
    nLayers = 5
    weightsReg = 1e-3

    # Build the encoder
    encoderInputs = keras.Input(shape=(nInputs),name = 'y_obs')
    for i in range(nLayers):
        if i == 0:
            x = layers.Dense(neurons, activation = 'tanh',kernel_regularizer = tf.keras.regularizers.L2(weightsReg))(encoderInputs)
        else:
            x = layers.Dense(neurons, activation = 'tanh',kernel_regularizer = tf.keras.regularizers.L2(weightsReg))(x)
    
    # Compute the posterior mean and covariance and uDraw
    meanPost = layers.Dense(nOutputs,name='meanPost',kernel_regularizer = tf.keras.regularizers.L2(weightsReg))(x)
    if covMatDiag == 0:
        vectorGammaCholesky = layers.Dense(nOutputs*(nOutputs+1)//2,name='diagCholesky',kernel_regularizer = tf.keras.regularizers.L2(weightsReg))(x)
        # Check fill triangular doc to see the filling of the lower triangular matrix
        inexactGammaCholesky = tfp.math.fill_triangular(vectorGammaCholesky) # You need to make the diagonal positive
        # Exponential of the diagonal part
        diagGammaCholesky = tf.math.exp(tf.linalg.diag_part(inexactGammaCholesky))
        gammaCholesky = tf.linalg.set_diag(inexactGammaCholesky,diagGammaCholesky)
    else:
        inexactGammaCholesky = layers.Dense(nOutputs,name='diagCholesky',kernel_regularizer = tf.keras.regularizers.L2(weightsReg))(x)
        gammaCholesky = tf.linalg.diag(tf.math.exp(inexactGammaCholesky))

    batch = tf.shape(meanPost)[0]
    epsilon = tf.random.normal(shape=(batch,nOutputs))
    uDraw = tf.math.add(meanPost,tf.linalg.matvec(gammaCholesky, epsilon))
    
    
    # Build the decoder
    for i in range(nLayers):
        if i == 0:
            x = layers.Dense(neurons, activation = 'tanh',kernel_regularizer = tf.keras.regularizers.L2(weightsReg))(uDraw)
        else:
            x = layers.Dense(neurons, activation = 'tanh',kernel_regularizer = tf.keras.regularizers.L2(weightsReg))(x)

    decoderOutputs = layers.Dense(nInputs,name='psi_d',kernel_regularizer = tf.keras.regularizers.L2(weightsReg))(x)
    
    combinedOutputs = layers.concatenate([meanPost[:,:,None], gammaCholesky], name='combined_output')
    
    encoder = keras.Model(inputs = [encoderInputs],outputs = [combinedOutputs,uDraw],name = 'encoder')
    decoder = keras.Model(inputs = [uDraw], outputs = [decoderOutputs], name = 'decoder')
    vae = keras.Model(inputs = [encoderInputs],outputs = [combinedOutputs, decoderOutputs, uDraw],name='vae')
    
    return vae, encoder, decoder


def lossDMandPrior(yTrue,yPred):
    # According to alpha \in (0,1) you have a family of methods
    #alpha = tf.constant(1.0)
    #alpha = tf.constant(0.75)
    alpha = tf.constant(0.5)
    #alpha = tf.constant(0.05)
    #alpha = tf.constant(0.001)
    const = tf.math.divide(tf.math.subtract(tf.constant(1.0),alpha),alpha)
    meanPost = yPred[:,:,0] # Mean of the estimated distribution for each sample of the batch
    gammaCholesky = yPred[:,:,1:] # Cholesky matrix of the covariance matrix of the estimated distribution for each sample of the batch
    
    # Loss terms
    # Data misfit
    normSamples = tf.math.reduce_sum(tf.math.multiply(tf.math.subtract(meanPost,yTrue),\
                                                      tf.reshape(tf.linalg.cholesky_solve(gammaCholesky,tf.reshape(tf.math.subtract(meanPost,yTrue),[yTrue.shape[0],yTrue.shape[1],1])),[yTrue.shape[0],yTrue.shape[1]])),\
                                                        axis = 1)
    dataMisfit = tf.math.multiply(const,normSamples)

    # Prior model
    trace = tf.linalg.trace(tf.linalg.matmul(tf.linalg.solve(gammaPriorScaled,gammaCholesky), gammaCholesky,transpose_b = True))
    normMeans = tf.math.reduce_sum(tf.math.multiply(tf.math.subtract(meanPost,meanPriorScaled),\
                                                      tf.reshape(tf.linalg.solve(gammaPriorScaled,tf.reshape(tf.math.subtract(meanPost,meanPriorScaled),[yTrue.shape[0],yTrue.shape[1],1])),[yTrue.shape[0],yTrue.shape[1]])),\
                                                        axis = 1)
    priorModel = tf.math.add(trace,normMeans)

    # Logaritm part (for stability is separated by the other terms)
    # (1-2*alpha)/alpha*Log|gammaPost| = (1-2*alpha)/alpha*2*log|gammaCholesky|
    logDetGammaPost = tf.math.multiply(tf.math.multiply(tf.math.subtract(const,tf.constant(1.0)),tf.constant(2.0)),tf.math.log(tf.math.reduce_prod(tf.linalg.diag_part(gammaCholesky),axis = 1)))
    
    return tf.math.reduce_mean(tf.math.add(tf.math.add(dataMisfit,priorModel),logDetGammaPost))

def lossLK(yTrue,yPred):
    # Likelihood model
    normPred = tf.math.reduce_sum(tf.math.multiply(tf.math.subtract(yTrue,yPred),\
        tf.reshape(tf.linalg.solve(gammaNoiseScaled,tf.reshape(tf.math.subtract(yTrue,yPred),[yTrue.shape[0],yTrue.shape[1],1])),[yTrue.shape[0],yTrue.shape[1]])),\
            axis = 1)
    return tf.math.reduce_mean(normPred)
    
def paramRmse(yTrue,yPred):
    # The divisions in the rescaling cancel each other out computing the RMSE
    meanPost = tf.math.subtract(yPred[:,:,0],tf.constant(scalerInput.min_,dtype=tf.float32))
    yTrueRescaled = tf.math.subtract(yTrue,tf.constant(scalerInput.min_,dtype=tf.float32))
    return tf.math.sqrt(tf.math.reduce_mean(tf.math.divide(tf.math.square(tf.math.subtract(yTrueRescaled,meanPost)),tf.math.square(yTrueRescaled))))
    
def outputRmse(yTrue,yPred):
    # The divisions in the rescaling cancel each other out computing the RMSE
    yPredRescaled = tf.math.subtract(yPred,tf.constant(scalerOutput.min_,dtype=tf.float32))
    yTrueRescaled = tf.math.subtract(yTrue,tf.constant(scalerOutput.min_,dtype=tf.float32))
    return tf.math.sqrt(tf.math.reduce_mean(tf.math.divide(tf.math.square(tf.math.subtract(yTrueRescaled,yPredRescaled)),tf.math.square(yTrueRescaled))))

# Choose if you want the covariance posterior matrix diagonal (1) or general (0)
covMatDiag = 0
# Choose if you want to save the model
save = 0
noise = 0.05
noisePath = 'noise005/'

# Load the dataset
dataPath = 'data/'
sobolFile = 'sobolIndicesTot.csv'


dimDataset = 100
batchSize = dimDataset//100 # The dataset must be divisible by 100!!!
#batchSize = 1 # For testing
# The dataset will be composed of the hypertuning set (dimHt), the training set (8dimHt) and the test set (dimHt) 
dimHt = int(dimDataset*0.1)

input = np.loadtxt(dataPath +'params9.txt', delimiter = ',')[:dimDataset,:]
output = np.loadtxt(dataPath +'outputs9.txt', delimiter = ',')[:dimDataset,:]

# Prior statistics
meanPrior = np.loadtxt(dataPath +'meanPrior9.txt', delimiter = ',')
gammaPrior = np.loadtxt(dataPath +'covPrior9.txt', delimiter = ',')

# These indices are related to the inputs according to GSA (queste linee non vi dovrebbero servire)
outputIndices = [0, 11, 12, 13, 31, 32, 33, 34]
output = output[:,outputIndices]

# Noise statistics
muNoise = np.zeros(output.shape[1])
gammaNoise = np.diag((noise*np.max(np.absolute(output),0))**2)
# Add noise to outputs
output = output+np.random.multivariate_normal(muNoise,gammaNoise,dimDataset)

# Scaling inputs and outputs in [-1,1]
# I consider only the hyperparameter tuning and training set for the scaling, removing the unseen test set
# In fact, except from here, I do not the hyperparameter tuning set 
inputScaled, scalerInput = scale(input,dimHt)
outputScaled, scalerOutput = scale(output,dimHt)

# Scaling of mean and covariance and repeating them for batchSize times to compute the loss function
meanPriorScaledNumpy = scalerInput.transform(meanPrior[None,:])[0,:]
meanPriorScaledCopies = np.repeat(meanPriorScaledNumpy[None,:],batchSize,axis=0)
meanPriorScaled = tf.convert_to_tensor(meanPriorScaledCopies, dtype = tf.float32)

gammaPriorScaledNumpy = np.multiply(gammaPrior,np.outer(scalerInput.scale_,scalerInput.scale_))
gammaPriorScaledCopies = np.repeat(gammaPriorScaledNumpy[None,:,:],batchSize,axis=0)
gammaPriorScaled = tf.convert_to_tensor(gammaPriorScaledCopies,dtype = tf.float32)

# Scaling of noise
gammaNoiseScaledNumpy = np.multiply(gammaNoise,np.outer(scalerOutput.scale_,scalerOutput.scale_))
gammaNoiseScaledCopies = np.repeat(gammaNoiseScaledNumpy[None,:,:],batchSize,axis=0)
gammaNoiseScaled = tf.convert_to_tensor(gammaNoiseScaledCopies,dtype = tf.float32)

# Define Keras model
# The outputs of the cardiovascular model are the input of the UQ-VAE
# Training the vae wiil train also the encoder and the decoder
vae, encoder, decoder = create_uqvae(outputScaled.shape[1],inputScaled.shape[1])

# Compile the Keras model
lr = 1e-5 # Change in the learning rate to improve the training stability and avoiding exploding gradients
encoder.compile(loss = [lossDMandPrior,None], optimizer = keras.optimizers.Adam(learning_rate=lr), metrics = [[paramRmse], [None]])
decoder.compile(loss = [lossLK], optimizer = keras.optimizers.Adam(learning_rate=lr), metrics = [[outputRmse]])
vae.compile(loss = [lossDMandPrior,lossLK,None], optimizer = keras.optimizers.Adam(learning_rate=lr), metrics = [[paramRmse], [outputRmse], [None]])

inputScaledTrain, inputScaledTest = inputScaled[dimHt:9*dimHt,:], inputScaled[9*dimHt:,:]
outputScaledTrain, outputScaledTest = outputScaled[dimHt:9*dimHt,:], outputScaled[9*dimHt:,:]

# Fit keras model on the dataset
# I stop the training if for 150 epochs the validation loss function does not improve and I take the weights returning the minimum val loss
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=150, mode = 'min',restore_best_weights=True)
# Use 100 for testing the code.
# In (inputScaledTrain,outputScaledTrain,inputScaledTrain) and (inputScaledTest,outputScaledTest,inputScaledTest) the last entry is useless,
# as there is not a third loss function. These are needed only preventing tensorflow from complaining as the VAE has three outputs
history = vae.fit(outputScaledTrain, (inputScaledTrain,outputScaledTrain,inputScaledTrain), epochs = 5000, batch_size = batchSize, \
                  validation_data=(outputScaledTest, (inputScaledTest,outputScaledTest,inputScaledTest)),callbacks=[callback],verbose = 1)

if save:
    encoder.save(dataPath+noisePath+'encoderTanVAE.keras')
    decoder.save(dataPath+noisePath+'decoderTanVAE.keras')
    vae.save(dataPath+noisePath+'vaeTanVAE.keras')

    np.savetxt(dataPath+noisePath+'TanTrainLoss' + '.txt',history.history['loss'],delimiter=',')
    np.savetxt(dataPath+noisePath+'TanValLoss' + '.txt',history.history['val_loss'],delimiter=',')

    joblib.dump(scalerInput, dataPath+noisePath+'scalerInputTanVAE.save') 
    joblib.dump(scalerOutput, dataPath+noisePath+'scalerOutputTanVAE.save') 

# Summarize history for loss
plt.figure(figsize=(10,10))
plt.rcParams.update({'font.size': 30})
ax = plt.plot(history.history['loss'],linewidth=4)
plt.plot(history.history['val_loss'],linewidth=4)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.gca().set_ylim(bottom=0)
plt.savefig('simulations/lossTanVAE.png')

"""
[predCombinedOutputs, predDecoderOutputs, predUDraw] = vae.predict(outputScaledTrain[0,:][None,:],verbose = 0)
# Compute the full covariance matrix
meanPost = scalerInput.inverse_transform(predCombinedOutputs[0,:,0][None,:])
gammaPost = np.multiply(np.outer(1/scalerInput.scale_,1/scalerInput.scale_),np.matmul(predCombinedOutputs[0,:,1:],np.transpose(predCombinedOutputs[0,:,1:])))
# Compute Pearson correlation coefficient
stdPost = np.sqrt(np.diag(gammaPost))
corrCoefPost = np.multiply(gammaPost,np.outer(1/stdPost,1/stdPost))
"""

"""
print('Posterior mean: ')
print(meanPost)
print('Posterior covariance matrix: ')
print(gammaPost)
print('Posterior correlation coefficient: ')
print(corrCoefPost)
"""

"""
# Compute an approximation of the upper bound of the loss functions
# Remember that the real bound in the paper of Thanh is computed when F is the true one and it is linear. Moreover, the bound is computed using mu_true
# In the linear case if we get to the minimum of the upper bound we have mu_post = mu_true
priorModelError = np.inner(predCombinedOutputs[0,:,0]-meanPriorScaledNumpy,slv(gammaPriorScaledNumpy,predCombinedOutputs[0,:,0]-meanPriorScaledNumpy,assume_a='pos'))
likelihoodError = np.inner(outputScaled[0,:]-decoder.predict(predCombinedOutputs[0,:,0][None,:],verbose = 0)[0,:],slv(gammaNoiseScaledNumpy,outputScaled[0,:]-decoder.predict(predCombinedOutputs[0,:,0][None,:],verbose = 0)[0,:],assume_a='pos'))
print('Loss upper bound: %f (alpha = 0.5)' %(2*meanPriorScaledNumpy.shape[0]+priorModelError+likelihoodError))
"""