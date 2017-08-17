How to Train Our Model?

Author: Xiangguo Sun

E-mail: sunxiangguo@seu.edu.cn

Although it is never an easy job to train a deep learning model, there are still some tricks you can use.


1. batch size

Generally speaking, a larger batch size will tell the model to go to a  more decisive direction, but it needs more time. On the other hand, a smaller batch size may lead to dramastic change and need more time to be stable.


2. overfited

You should split your raw data into 3parts: train data, validation data,and test data.

Sometimes you may get good results on test data, but that doesn't means your model parameters are good. For example, you may train your model as follows:

Train on 181 samples, validate on 78 samples

Epoch 1/100

9s - loss: 0.7635 - accuracy: 0.4530 - val_loss: 0.7072 - accuracy: 0.4615

Epoch 2/100

9s - loss: 0.6881 - accuracy: 0.5414 - val_loss: 0.7005 - accuracy: 0.4615

Epoch 3/100

9s - loss: 0.6762 - accuracy: 0.5967 - val_loss: 0.6810 - accuracy: 0.5385

Epoch 4/100

9s - loss: 0.6901 - accuracy: 0.5359 - val_loss: 0.7335 - accuracy: 0.4615

Epoch 5/100

9s - loss: 0.6979 - accuracy: 0.5580 - val_loss: 0.6711 - accuracy: 0.5897

Epoch 6/100

9s - loss: 0.6332 - accuracy: 0.6906 - val_loss: 0.6797 - accuracy: 0.5641

Epoch 7/100

9s - loss: 0.6240 - accuracy: 0.7238 - val_loss: 0.6884 - accuracy: 0.5769

Epoch 8/100

9s - loss: 0.6056 - accuracy: 0.6630 - val_loss: 0.6678 - accuracy: 0.5128

Epoch 9/100

9s - loss: 0.5853 - accuracy: 0.7569 - val_loss: 0.6936 - accuracy: 0.5513

Epoch 10/100

9s - loss: 0.5796 - accuracy: 0.7127 - val_loss: 0.6436 - accuracy: 0.6154

Epoch 11/100

9s - loss: 0.5341 - accuracy: 0.8122 - val_loss: 0.6872 - accuracy: 0.5769

Epoch 12/100

9s - loss: 0.4928 - accuracy: 0.7735 - val_loss: 0.6331 - accuracy: 0.6538

Epoch 13/100

9s - loss: 0.4851 - accuracy: 0.7735 - val_loss: 0.6655 - accuracy: 0.6154

Epoch 14/100

9s - loss: 0.4195 - accuracy: 0.8398 - val_loss: 0.7325 - accuracy: 0.6154

Epoch 15/100

9s - loss: 0.4572 - accuracy: 0.7459 - val_loss: 0.7757 - accuracy: 0.5769

Epoch 16/100

9s - loss: 0.3764 - accuracy: 0.8287 - val_loss: 0.7636 - accuracy: 0.5897

train process done!!

('micro Precision score for classification model - ', 0.62962962962962965)

('macro Precision score for classification model - ', 0.63174603174603172)

Although the final results on text data is pretty good, but trace your training history, you will find that
the gap between the validation loss and training loss is too big(from initial around 0.02 to the last around 0.4) ,
which means your training process may exsit overfited phenomenon.

In addition, the number of training data seems to be so small:

Train on 181 samples, validate on 78 samples

If possible, try to enlarge your training data.

You should also try to enlarge the rate in the dropout layer

if posible, try to update your positive samples and negtive samples in your data.

last but not least, you can change other parameters and nodes numbers in your model to fit your data more well.


Here is a  good training instance:

Epoch 1/100

10s - loss: 0.7842 - categorical_accuracy: 0.4807 - val_loss: 0.6818 - val_categorical_accuracy: 0.6538

Epoch 2/100

9s - loss: 0.6779 - categorical_accuracy: 0.5912 - val_loss: 0.6934 - val_categorical_accuracy: 0.5000

Epoch 3/100

9s - loss: 0.6793 - categorical_accuracy: 0.5304 - val_loss: 0.6918 - val_categorical_accuracy: 0.5385

Epoch 4/100

9s - loss: 0.6989 - categorical_accuracy: 0.5414 - val_loss: 0.7113 - val_categorical_accuracy: 0.4359

Epoch 5/100

9s - loss: 0.7141 - categorical_accuracy: 0.5304 - val_loss: 0.6870 - val_categorical_accuracy: 0.5256

Epoch 6/100

9s - loss: 0.6314 - categorical_accuracy: 0.6685 - val_loss: 0.6854 - val_categorical_accuracy: 0.5128

train process done!!

('micro Precision score for classification model - ', 0.62962962962962965)

('macro Precision score for classification model - ', 0.61852407129768439)

trace the training history, we can find that the gap between val_loss and train_loss is about 0.03,
more excitingly,the final result on test data is also good.




Best Wishes
Good luck!
