import pandas as pd
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

#Chargement du modèle
model = TFT5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5TokenizerFast.from_pretrained("t5-base")
optimizer = tensorflow.keras.optimizers.Adam()

#Taille des batchs
batch_size=32
num_of_batches=math.floor(len(df_train)/batch_size)
             
             
#Entrainement
epoch=2
for k in range(epoch):
    print('Running epoch: {}/{}'.format(k+1,epoch))
    running_loss=0
    
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
    
        # Actualisation des poids
        gradients = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        
        running_loss+=loss_value
        writer("\r Batch number : " + str(i+1) + "/" + str(num_of_batches))
    
    running_loss=running_loss/int(num_of_batches)
    print('Epoch: {} , Running loss: {}'.format(k+1,running_loss))

#Sauvegarde du modèle
model.save_pretrained("./model")
