import numpy as np
import utils
import typing
import math
np.random.seed(1)

SIGMOID = 1
SOFTMAX = 2
class Layer:
    activation_function : int
    w = np.array(0)
    z = np.array(0)
    a = np.array(0)
    def __init__(self, activation_function=0, shape=(0)):
        self.activation_function = activation_function
        self.w = np.random.uniform(-1,1,shape)

def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))

def softmax(a: np.ndarray):
    a_exp = np.exp(a)
    return a_exp / a_exp.sum(axis=1, keepdims=True)
'''
def Gamma_d(z: np.ndarray):
    return df = sigmoid(z)*(1-sigmoid(z))
'''
def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785]
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"

    global mean, std
    X = np.insert(X, 0, 1, axis=1)
    
    X = (X-(mean))/(std)
    X = np.divide(X,255)
    return X


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    ce = targets * np.log(outputs)
    ce = np.sum(ce, axis=0)
    ce = np.sum(ce)
    ce = -1*ce
    return ce


class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            w = np.random.uniform(-1,1,(w_shape))            
            self.ws.append(w)
            prev = size
        self.grads = [0 for i in range(len(self.ws))]

        #define a_i and a_j
        self.z_j = []
        self.a_j = []


    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        self.z_j = np.matmul(X,self.ws[0]) #(100,64)
        self.a_j = sigmoid(self.z_j) #(100,64)
        z_k = np.matmul(self.a_j,self.ws[1]) #(100,10)
        output = softmax(z_k) #(100,10)
        return output

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer

        d_k = -(targets - outputs) #(100x10)
        grad_1 = np.matmul(self.a_j.T, d_k)
    
        df = self.a_j*(1-self.a_j) #(100x64)
        temp = self.ws[1].dot(d_k.T) #(64x100)
        d_j = df.T*temp #(64x100)
        grad_0 = np.matmul(X.T, d_j.T)

        self.grads = [grad_0, grad_1]


        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    encoded = np.zeros((Y.size, num_classes))
    encoded[np.arange(Y.size), Y.squeeze()] = 1
    return encoded


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist(0.1)
    mean = np.mean(X_train)
    std = np.std(X_train)
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)
    gradient_approximation_test(model, X_train, Y_train)
