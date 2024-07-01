import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os # to use operating system dependent functionality
import librosa # to extract speech features
import wave # read and write WAV files

# MLP Classifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# LSTM Classifier
import keras
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import *
def extract_mfcc(wav_file_name):
    #This function extracts mfcc features and obtain the mean of each dimension
    #Input : path_to_wav_file
    #Output: mfcc_features'''
    y, sr = librosa.load(wav_file_name)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,axis=0)
    
    return mfccs
##### load radvess speech data #####

def training_the_model():
    radvess_speech_labels = [] # to save extracted label/file
    ravdess_speech_data = [] # to save extracted features/file
    for dirname, _, filenames in os.walk('ravdess-emotional-speech-audio/'):
        for filename in filenames:
            #print(os.path.join(dirname, filename))
            radvess_speech_labels.append(int(filename[7:8]) - 1) # the index 7 and 8 of the file name represent the emotion label
            wav_file_name = os.path.join(dirname, filename)
            ravdess_speech_data.append(extract_mfcc(wav_file_name)) # extract MFCC features/file
            
    print("Finish Loading the Dataset")
    #### convert data and label to array
    ravdess_speech_data_array = np.asarray(ravdess_speech_data) # convert the input to an array
    ravdess_speech_label_array = np.array(radvess_speech_labels)
    ravdess_speech_label_array.shape # get tuple of array dimensions

    #### make categorical labels
    labels_categorical = to_categorical(ravdess_speech_label_array) # converts a class vector (integers) to binary class matrix
    labels_categorical.shape
    ravdess_speech_data_array.shape
    x_train,x_test,y_train,y_test= train_test_split(np.array(ravdess_speech_data_array),labels_categorical, test_size=0.20, random_state=9)
    # Split the training, validating, and testing sets
    number_of_samples = ravdess_speech_data_array.shape[0]
    training_samples = int(number_of_samples * 0.8)
    validation_samples = int(number_of_samples * 0.1)
    test_samples = int(number_of_samples * 0.1)
    # Define the LSTM model
    def create_model_LSTM():
        model = Sequential()
        model.add(LSTM(128, return_sequences=False, input_shape=(40, 1)))
        model.add(Dense(64))
        model.add(Dropout(0.4))
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(Dropout(0.4))
        model.add(Activation('relu'))
        model.add(Dense(8))
        model.add(Activation('softmax'))
        
        # Configures the model for training
        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        return model
    w = np.expand_dims(ravdess_speech_data_array[:training_samples],-1)
    w.shape
    ### train using LSTM model
    model_A = create_model_LSTM()
    history = model_A.fit(np.expand_dims(ravdess_speech_data_array[:training_samples],-1), labels_categorical[:training_samples], validation_data=(np.expand_dims(ravdess_speech_data_array[training_samples:training_samples+validation_samples], -1), labels_categorical[training_samples:training_samples+validation_samples]), epochs=121, shuffle=True)
    ### loss plots using LSTM model
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(loss) + 1)                                       

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    ### evaluate using model A
    model_A.evaluate(np.expand_dims(ravdess_speech_data_array[training_samples + validation_samples:], -1), labels_categorical[training_samples + validation_samples:])

from keras.models import load_model

def create_model():
    model = Sequential()
    model.add(LSTM(128, return_sequences=False, input_shape=(40, 1)))
    model.add(Dense(64))
    model.add(Dropout(0.4))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Dropout(0.4))
    model.add(Activation('relu'))
    model.add(Dense(8))
    model.add(Activation('softmax'))
    return model

def load_trained_model():
   model = create_model()
   model.load_weights("mymodel.h5")
   return model

loaded_model = load_trained_model()

emotions={1 : 'neutral', 2 : 'calm', 3 : 'happy', 4 : 'sad', 5 : 'angry', 6 : 'fearful', 7 : 'disgust', 8 : 'surprised'}

def predict(wav_filepath):
  test_point=extract_mfcc(wav_filepath)
  test_point=np.reshape(test_point,newshape=(1,40,1))
  predictions=loaded_model.predict(test_point)
  return emotions[np.argmax(predictions[0])+1]

# model_A.save('mymodel.h5')
# modelc=tf.keras.models.load_model('mymodel.h5')
# predict('/akhil.wav')
# predict('/vd.wav')