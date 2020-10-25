import numpy as np
import tensorflow as tf

tf.keras.backend.set_floatx('float64')

#custom model class
class NN(tf.keras.Model):

    def __init__(self, input_size, h_layers, h_layers_size, output_size):

        super(NN, self).__init__()
        self.flatten = tf.keras.layers.Flatten(input_shape = input_size)
        self.l1 = tf.keras.layers.Dense(h_layers_size[0], name = 'l1')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.l2 = tf.keras.layers.Dense(output_size, name = 'l2')

    def __call__(self, input, training = False):

        features = self.flatten(input)
        features = tf.nn.relu(self.l1(features))
        if training:
            features = self.dropout(features, training = training)
        features = tf.nn.softmax(self.l2(features))

        return features

def loss(model, x, y):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
    y_pred = model(x, training= True)
    return loss_object(y_true = y, y_pred = y_pred)

def grad(model, x, y):
    with tf.GradientTape() as tape:
        loss_out = loss(model, x, y)
    return loss_out, tape.gradient(loss_out, model.trainable_variables)



#custom training
def training(model, train_dataset, epochs, optimizer):
    train_loss = []
    train_accuracy = []

    for epoch in range(epochs):
        loss_avg = tf.keras.metrics.Mean()
        accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        for x, y in train_dataset:
            losses, gradients = grad(model, x, y)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            loss_avg(losses)
            accuracy(y, model(x))
            print("Epoch: {:03d} Loss: {:.3f} Accuracy: {:.3%}".format(epoch + 1, loss_avg.result(), accuracy.result()))

        train_loss.append(loss_avg.result())
        train_accuracy.append(accuracy.result())

        if epoch % 1 == 0:
            print("Epoch: {:03d} Loss: {:.3f} Accuracy: {:.3%}".format(epoch + 1, loss_avg.result(), accuracy.result()))



# input pipeline
buffer_size = 1024
batch_size = 600
#train_dataset = tfds.load(name = 'fashion_mnist', split = 'train')
#test_dataset = tfds.load(name = 'fashion_mnist', split = 'test')
train, test = tf.keras.datasets.fashion_mnist.load_data()
x_train, y_train = train
x_test, y_test = test
x_train = x_train/255
x_test = x_test/255
#max_train_samples = 20
#x_train = x_train[:max_train_samples]
#y_train = y_train[:max_train_samples]

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size).repeat(50)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
input_shape = (x_train.shape[1], x_train.shape[2])


model = NN(input_size= input_shape, h_layers = 2, h_layers_size = [128], output_size = 10)
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
epochs = 10

training(model, train_dataset, epochs, optimizer)

#model.evaluate(x_test, y_test)