import pandas as pd
import numpy as np
import math
import tensorflow
from transformers import T5TokenizerFast, TFT5ForConditionalGeneration
import sys
writer = sys.stdout.write

#Chargement des données, récupération des données d'entrainement
data = pd.read_csv('New_dataset_time.csv',index_col=0)
n = int(0.8*data.shape[0])
df_train = data.iloc[:n,1:]
df_label = data.iloc[:n,0:1]
df_validation = data.iloc[n:,1:]
df_lab_valid = data.iloc[n:,0:1]

#Chargement du modèle
model = TFT5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5TokenizerFast.from_pretrained("t5-base")
optimizer = tensorflow.keras.optimizers.Adam()

#Taille des batchs
batch_size=32
num_of_batches=math.floor(len(df_train)/batch_size)
num_of_batches_val = math.floor(len(df_validation)/batch_size)
             
             
#Entrainement
epoch=1
delta = 1
validation_loss = 10
training_loss_history = []
validation_loss_history = []
mean_training_loss_history = []
mean_validation_loss_history = []
while delta > 0.01:
    print('Running epoch: {}'.format(epoch))
    training_loss=0
    list_loss = []
    for i in range(num_of_batches):
        #Construction du batch
        inputbatch=[]
        labelbatch=[]
        new_df=df_train[i*batch_size:i*batch_size+batch_size]
        new_df2=df_label[i*batch_size:i*batch_size+batch_size]
        for indx,row in new_df.iterrows():
            input = str(row) 
            #print(len(input))
            inputbatch.append(input)
        for indx,row in new_df2.iterrows():
            labels = row['txt']
            labelbatch.append(labels)
        inputbatch=tokenizer.batch_encode_plus(inputbatch,padding=True,max_length=900,return_tensors='tf')["input_ids"]
        labelbatch=tokenizer.batch_encode_plus(labelbatch,padding=True,max_length=900,return_tensors="tf") ["input_ids"]
        #Entrainement sur le batch
        with tensorflow.GradientTape() as tape:
            outputs = model(input_ids=inputbatch, labels=labelbatch)
            # Compute the loss value for this batch.
            loss_value = sum(outputs.loss)/len(outputs.loss)
            list_loss.append(loss_value)
        # Actualisation des poids
        gradients = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        
        training_loss+=loss_value
        writer("\r Batch number : " + str(i+1) + "/" + str(num_of_batches))
    
    training_loss=training_loss/int(num_of_batches)
    print('Epoch: {} , Running loss: {}'.format(k,training_loss))

    training_loss_history.append(list_loss)
    mean_training_loss_history.append(np.mean(list_loss))

    #Calcul validation_loss
    list_loss2 = []
    for i in range(num_of_batches_val):
        inputbatch=[]
        labelbatch=[]
        new_df=df_validation[i*batch_size:i*batch_size+batch_size]
        new_df2=df_lab_valid[i*batch_size:i*batch_size+batch_size]
        for indx,row in new_df.iterrows():
            input = str(row)
            inputbatch.append(input)
        for indx,row in new_df2.iterrows():
            labels = row['txt']
            labelbatch.append(labels)
        inputbatch=tokenizer.batch_encode_plus(inputbatch,padding=True,max_length=900,return_tensors='tf')["input_ids"]
        labelbatch=tokenizer.batch_encode_plus(labelbatch,padding=True,max_length=900,return_tensors="tf") ["input_ids"]
        with tensorflow.GradientTape() as tape:
            outputs = model(input_ids=inputbatch, labels=labelbatch)
            # Compute the loss value for this batch.
            loss_value = sum(outputs.loss)/len(outputs.loss)
            list_loss2.append(loss_value)

    delta = validation_loss - np.mean(list_loss2)
    validation_loss = np.mean(list_loss2)
    validation_loss_history.append(list_loss2)
    mean_validation_loss_history.append(validation_loss)
    if delta > 0.01:
        #Sauvegarde du modèle
        model.save_pretrained("./model")
    epoch += 1

data1 = pd.DataFrame(training_loss_history)
data2 = pd.DataFrame(validation_loss_history)
data3 = pd.DataFrame(mean_training_loss_history)
data4 = pd.DataFrame(mean_validation_loss_history)
data1.to_csv('training_loss_history.csv', index=False)
data2.to_csv('validation_loss_history.csv', index=False)
data3.to_csv('mean_training_loss_history.csv', index=False)
data4.to_csv('mean_validation_loss_history.csv', index=False)
