import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import cross_entropy_loss, BinaryModel, pre_process_images
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: BinaryModel) -> float: 
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 1]
        model: model of class BinaryModel
    Returns:
        Accuracy (float)
    """
    number_of_predictions = X.shape[0]
    number_of_rights = 0
    y_hat = model.forward(X)

    for index in range (0, number_of_predictions):
        if y_hat[index] >= 0.5:
            y_hat[index] = 1
        else: 
            y_hat[index] = 0

        if y_hat[index] == targets[index]:
            number_of_rights += 1
    # Task 2c
    accuracy = number_of_rights/number_of_predictions
    return accuracy


def train(
        num_epochs: int,
        learning_rate: float,
        batch_size: int,
        l2_reg_lambda: float # Task 3 hyperparameter. Can be ignored before this.
        ):
    """
        Function that implements logistic regression through mini-batch
        gradient descent for the given hyperparameters
    """
    print("Du er inne i train")
    global X_train, X_val, X_test
    # Utility variables
    num_batches_per_epoch = X_train.shape[0] // batch_size
    num_steps_per_val = num_batches_per_epoch // 5
    train_loss = {}
    val_loss = {}
    train_accuracy = {}
    val_accuracy = {}
    model = BinaryModel(l2_reg_lambda)
    if X_train.shape[1]==784:
        X_train = pre_process_images(X_train)
    if X_test.shape[1]==784:
        X_test = pre_process_images(X_test)
    if X_val.shape[1]==784:
        X_val = pre_process_images(X_val)

    global_step = 0
    for epoch in range(num_epochs):
        for step in range(num_batches_per_epoch):
            # Select our mini-batch of images / labels
            start = step * batch_size
            end = start + batch_size
            X_batch, Y_batch = X_train[start:end], Y_train[start:end]

            y_hat = model.forward(X_batch)
            
            model.backward(X_batch, y_hat, Y_batch)
            model.w += -1*learning_rate*model.grad

            # Track training loss continuously
            _train_loss = cross_entropy_loss(Y_batch, y_hat)
            train_loss[global_step] = _train_loss
            # Track validation loss / accuracy every time we progress 20% through the dataset
            if global_step % num_steps_per_val == 0:
                _val_loss = cross_entropy_loss(Y_val, model.forward(X_val))
                val_loss[global_step] = _val_loss

                train_accuracy[global_step] = calculate_accuracy(
                    X_train, Y_train, model)
                val_accuracy[global_step] = calculate_accuracy(
                    X_val, Y_val, model)

            global_step += 1
    return model, train_loss, val_loss, train_accuracy, val_accuracy


# Load dataset
category1, category2 = 2, 3
validation_percentage = 0.1
X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.load_binary_dataset(
    category1, category2, validation_percentage)

# hyperparameters
num_epochs = 20
learning_rate = 0.2
batch_size = 128
l2_reg_lambda = 0.1
model, train_loss, val_loss, train_accuracy, val_accuracy = train(
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    l2_reg_lambda=1)
print("1 modell")
model1, train_loss1, val_loss1, train_accuracy1, val_accuracy1 = train(
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    l2_reg_lambda=0.1)
print("2 modeller")
model2, train_loss2, val_loss2, train_accuracy2, val_accuracy2 = train(
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    l2_reg_lambda=0.01)
model3, train_loss3, val_loss3, train_accuracy3, val_accuracy3 = train(
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    l2_reg_lambda=0.001)

print("Final Train Cross Entropy Loss:",
      cross_entropy_loss(Y_train, model.forward(X_train)))
print("Final  Test Entropy Loss:",
      cross_entropy_loss(Y_test, model.forward(X_test)))
print("Final Validation Cross Entropy Loss:",
      cross_entropy_loss(Y_val, model.forward(X_val)))


print("Train accuracy:", calculate_accuracy(X_train, Y_train, model))
print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model))
print("Test accuracy:", calculate_accuracy(X_test, Y_test, model))

'''
# Plot loss
plt.figure(1)
plt.ylim([0., .4]) 
utils.plot_loss(train_loss, "Training Loss")
utils.plot_loss(val_loss, "Validation Loss")
plt.legend()
plt.savefig("binary_train_loss.png")
plt.show()


# Plot accuracy
plt.figure(2)
plt.ylim([0.93, .99])
utils.plot_loss(train_accuracy, "Training Accuracy")
utils.plot_loss(val_accuracy, "Validation Accuracy")
plt.legend()
plt.savefig("binary_train_accuracy.png")
plt.show()
'''
plt.figure(1)
plt.ylim([0.93, .99])
utils.plot_loss(train_accuracy, "Training Accuracy, l = 1")
utils.plot_loss(val_accuracy, "Validation Accuracy, l = 1")
utils.plot_loss(train_accuracy1, "Training Accuracy, l = 0.1")
utils.plot_loss(val_accuracy1, "Validation Accuracy, l = 0.1")
utils.plot_loss(train_accuracy2, "Training Accuracy, l = 0.01")
utils.plot_loss(val_accuracy2, "Validation Accuracy, l = 0.01")
utils.plot_loss(train_accuracy3, "Training Accuracy, l = 0.001")
utils.plot_loss(val_accuracy3, "Validation Accuracy, l = 0.001")
plt.legend()
plt.savefig("binary_train_accuracy.png")
plt.show()