import math
import tensorflow as tf
from transformers import T5TokenizerFast, TFT5ForConditionalGeneration
from progressbar import progressbar as pb

from dataset_creation.utils import load_dataset

data = load_dataset("model_creation/final_dataset.csv")

n = len(data)

df_input = data.iloc[:, 1:]
df_label = data.iloc[:, 0:1]

model = TFT5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5TokenizerFast.from_pretrained("t5-base")
optimizer = tf.keras.optimizers.Adam()

batch_size = 32
num_of_batches = math.floor(n / batch_size)

label = [df_label.values[i][0] for i in range(len(data))]
label_encoded = tokenizer.batch_encode_plus(
    label, padding=True, max_length=900, return_tensors="tf"
)["input_ids"]

inputbatch = [
    "|".join(df_input.iloc[i].dropna().values.tolist()) for i in range(len(data))
]
input_encoded = tokenizer.batch_encode_plus(
    inputbatch, padding=True, max_length=900, return_tensors="tf"
)["input_ids"]

epoch = 2
for k in range(epoch):

    for i in pb(range(num_of_batches)):
        input = input_encoded[i * batch_size : i * batch_size + batch_size]
        output = label_encoded[i * batch_size : i * batch_size + batch_size]

        with tf.GradientTape() as tape:
            outputs = model(input_ids=input, labels=output)

            loss_value = sum(outputs.loss) / len(outputs.loss)

        gradients = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

model.save_pretrained("../model")
