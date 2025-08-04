import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt


(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split= ['train','test'],
    shuffle_files = True,
    as_supervised = True,
    with_info = True,

)

# for preprocessing
def normalize_img(image,labels) :
  return tf.cast(image, tf.float32)/255, labels

ds_train = ds_train.map(normalize_img, num_parallel_calls= tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)   # weights for each interation
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)


# rating perform
ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

# data modeling - train

## sequential - set up
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),
    tf.keras.layers.Dense(128, activation = 'relu') , # define algo to model learning
    tf.keras.layers.Dense(10)   # 10 node for 10 different data's types


])


## compiling - configuring how to model learning

model.compile(
    # update weights
    optimizer = tf.keras.optimizers.Adam(0.001),
    # calculate loss -- LOTS OF DOCS NOT NOTICE YET
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

    # Monitoring model's training -- LOTS OF DOCS NOT NOTICE YET
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],



)


model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)



# preview

def predictions(model, ds_test, n_samples = 10) :

  ## first stage : get first pair
  ### logic : next(iter()) -> get the first interaction
  ###         seperation from 128 elements to 1 element by unbatch and continue batch 10 elements

  images, labels = next(iter(ds_test.unbatch().batch(n_samples)))

  ### raw prediction
  logits = model.predict(images)
  ### output prediction
  predictions = tf.argmax(logits, axis = 1)  # argmax = choose the most correctly



  ## second stage : draw picture and result
  plt.figure(figsize=(15, 5))
  for i in range(n_samples):
      plt.subplot(2, 5, i + 1)
      ### show ảnh
      plt.imshow(images[i], cmap='gray')

      ### attaching title
      plt.title(f"Pred: {predictions[i].numpy()}\nLabel: {labels[i].numpy()}",
                  color="green" if predictions[i] == labels[i] else "red")
      plt.axis('off')
  plt.tight_layout()
  plt.show()


predictions(model, ds_test)




