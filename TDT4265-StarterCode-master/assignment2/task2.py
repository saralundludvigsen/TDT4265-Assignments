import numpy as np
import utils
import matplotlib.pyplot as plt
import typing
from task2a import cross_entropy_loss, SoftmaxModel, one_hot_encode, pre_process_images
np.random.seed(0)

def shuffle_in_unison(a, b):
    state = np.random.get_state()
    a = np.random.shuffle(a)
    np.random.set_state(state)
    b = np.random.shuffle(b)

def calculate_accuracy(X: np.ndarray, targets: np.ndarray,
                       model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    number_of_predictions = X.shape[0]
    number_of_rights = 0
    y_hat = model.forward(X)
    for i in range (0, number_of_predictions):
        y_hat[i] = np.around(y_hat[i])
        if np.array_equal(y_hat[i],targets[i]):
            number_of_rights += 1
    accuracy = number_of_rights/number_of_predictions
    return accuracy


def train(
        model: SoftmaxModel,
        datasets: typing.List[np.ndarray],
        num_epochs: int,
        learning_rate: float,
        batch_size: int,
        # Task 3 hyperparameters,
        use_shuffle: bool,
        use_momentum: bool,
        use_early_stopping: bool,
        momentum_gamma: float):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = datasets

    # Utility variables
    num_batches_per_epoch = X_train.shape[0] // batch_size
    num_steps_per_val = num_batches_per_epoch // 5
    # Tracking variables to track loss / accuracy
    train_loss = {}
    val_loss = {}
    train_accuracy = {}
    val_accuracy = {}
    momentum = [0 for i in range(len(model.grads))]

    #Variables used for early stopping
    mean_val_loss = []
    list_val_losses = []

    global_loss_counter = 2
    global_step = 0
    for epoch in range(num_epochs):
        # Shuffling before next epoch
        if use_shuffle == True:
            shuffle_in_unison(X_train, Y_train)
        for step in range(num_batches_per_epoch):
            start = step * batch_size
            end = start + batch_size
            X_batch, Y_batch = X_train[start:end], Y_train[start:end]

            y_hat = model.forward(X_batch)
            model.backward(X_batch, y_hat, Y_batch)

            if use_momentum == True:
                momentum[0] = (1-momentum_gamma)*model.grads[0] + momentum_gamma*momentum[0]
                momentum[1] = (1-momentum_gamma)*model.grads[1] + momentum_gamma*momentum[1]
                model.ws[0] += -1*learning_rate*(momentum[0])
                model.ws[1] += -1*learning_rate*(momentum[1])
            else:
                model.ws[0] += -1*learning_rate*model.grads[0]
                model.ws[1] += -1*learning_rate*model.grads[1]

            # Track train / validation loss / accuracy
            # every time we progress 20% through the dataset
            if (global_step % num_steps_per_val) == 0:
                _val_loss = cross_entropy_loss(Y_val, model.forward(X_val))
                val_loss[global_step] = _val_loss

                _train_loss = cross_entropy_loss(Y_batch, y_hat)
                train_loss[global_step] = _train_loss

                train_accuracy[global_step] = calculate_accuracy(
                    X_train, Y_train, model)
                val_accuracy[global_step] = calculate_accuracy(
                    X_val, Y_val, model)
                
                #Early stopping
                if use_early_stopping == True:
                    list_val_losses.append(_val_loss)
                    if global_loss_counter % 5 == 0:
                        mean_val_loss.append(np.mean(list_val_losses))
                        list_val_losses = [] 
                        if global_loss_counter % 10 == 0:
                            if mean_val_loss[0] < mean_val_loss[1]:
                                return model, train_loss, val_loss, train_accuracy, val_accuracy
                            mean_val_loss = []
                    global_loss_counter += 1

            global_step += 1
    return model, train_loss, val_loss, train_accuracy, val_accuracy


if __name__ == "__main__":
    # Load dataset
    validation_percentage = 0.2
    X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.load_full_mnist(
        validation_percentage)
    mean = np.mean(X_train)
    std = np.std(X_train)
    X_train = pre_process_images(X_train, mean, std)
    X_test = pre_process_images(X_test, mean, std)
    X_val = pre_process_images(X_val, mean, std)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)
    Y_test = one_hot_encode(Y_test, 10)
    # Hyperparameters
    num_epochs = 20
    learning_rate = .1
    batch_size = 32
    neurons_per_layer_1 = [64, 10]
    neurons_per_layer_2 = [59, 59, 10]
    momentum_gamma = .9  # Task 3 hyperparameter

    # Settings for task 3. Keep all to false for task 2.
    use_shuffle = False
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = False
    use_early_stopping = False

    model = SoftmaxModel(
        neurons_per_layer_1,
        use_improved_sigmoid,
        use_improved_weight_init)
    model, train_loss, val_loss, train_accuracy, val_accuracy = train(
        model,
        [X_train, Y_train, X_val, Y_val, X_test, Y_test],
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        use_shuffle=use_shuffle,
        use_momentum=use_momentum,
        use_early_stopping=use_early_stopping,
        momentum_gamma=momentum_gamma)
    print("Model 0")
    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Test Cross Entropy Loss:",
          cross_entropy_loss(Y_test, model.forward(X_test)))

    print("Final Train accuracy:",
          calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:",
          calculate_accuracy(X_val, Y_val, model))
    print("Final Test accuracy:",
          calculate_accuracy(X_test, Y_test, model))    
    
    # 2nd plot
    use_shuffle = False
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = False
    use_early_stopping = False

    model1 = SoftmaxModel(
        neurons_per_layer_2,
        use_improved_sigmoid,
        use_improved_weight_init)
    model1, train_loss1, val_loss1, train_accuracy1, val_accuracy1 = train(
        model1,
        [X_train, Y_train, X_val, Y_val, X_test, Y_test],
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        use_shuffle=use_shuffle,
        use_momentum=use_momentum,
        use_early_stopping=use_early_stopping,
        momentum_gamma=momentum_gamma)
    print("Model 1")
    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model1.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model1.forward(X_val)))
    print("Final Test Cross Entropy Loss:",
          cross_entropy_loss(Y_test, model1.forward(X_test)))

    print("Final Train accuracy:",
          calculate_accuracy(X_train, Y_train, model1))
    print("Final Validation accuracy:",
          calculate_accuracy(X_val, Y_val, model1))
    print("Final Test accuracy:",
          calculate_accuracy(X_test, Y_test, model1))

    # Plot loss
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.ylim([0.0, .6])
    utils.plot_loss(train_loss, "Training Loss 1 hidden layer")
    utils.plot_loss(val_loss, "Validation Loss 1 hidden layer")
    utils.plot_loss(train_loss1, "Training Loss 2 hidden layers")
    utils.plot_loss(val_loss1, "Validation Loss 2 hidden layers")
    
    plt.xlabel("Number of gradient steps")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Task 4d, Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    # Plot accuracy
    plt.ylim([0.75, 1.0])
    utils.plot_loss(train_accuracy, "Training Accuracy 1 hidden layer")
    utils.plot_loss(val_accuracy, "Validation Accuracy 1 hidden layer")
    utils.plot_loss(train_accuracy1, "Training Accuracy 2 hidden layers")
    utils.plot_loss(val_accuracy1, "Validation Accuracy 2 hidden layers")
    plt.title("Task 4d, Accuracy")
    plt.legend()
    plt.xlabel("Number of gradient steps")
    plt.ylabel("Accuracy")
    plt.savefig("A2_4d.png")
    plt.show()
