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

def improved_sigmoid(x: np.ndarray):
    return 1.7159*np.tanh((2/3)*x)

def improved_sigmoid_derivative(x: np.ndarray):
    return 1.7159*2.0 / (3.0*(np.cosh(((2.0/3.0)*x))**2))

def softmax(a: np.ndarray):
    a_exp = np.exp(a)
    return a_exp / a_exp.sum(axis=1, keepdims=True)
'''
def Gamma_d(z: np.ndarray):
    return df = sigmoid(z)*(1-sigmoid(z))
'''
def pre_process_images(X: np.ndarray, mean: float, std:float):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785]
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"

    X = np.insert(X, 0, 1, axis=1)
    
    X = (X-(mean))/(std)
    #X = np.divide(X,255)
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
    ce = -(1/(targets.shape[0]))*ce
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
        self.grads = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            grad_shape = (prev, size)
            grad = np.random.uniform(-1,1,(grad_shape))
            if use_improved_weight_init == False:
                w = np.random.uniform(-1,1,(w_shape))        
            else: 
                w = np.random.normal(0, 1/(np.sqrt(w_shape[1])), w_shape)    
            self.ws.append(w)
            self.grads.append(grad)

            prev = size

        #define z, a and d
        self.z = [[None] for i in range(len(self.neurons_per_layer))]
        self.a = [[None] for i in range(len(self.neurons_per_layer)+1)]
        self.d = [[None] for i in range(len(self.neurons_per_layer))]



    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        self.a[0] = X
        self.z[0] = np.matmul(self.a[0],self.ws[0])
        for i in range(1,len(self.neurons_per_layer),1):
            if self.use_improved_sigmoid == True:
                self.a[i] = improved_sigmoid(self.z[i-1])
                self.z[i] = np.matmul(self.a[i],self.ws[i])
            else:
                self.a[i] = sigmoid(self.z[i-1])
                self.z[i] = np.matmul(self.a[i],self.ws[i])

        self.a[-1] = softmax(self.z[i])
        output = self.a[-1]
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

        self.d[-1] = -(targets - outputs)
        self.grads[-1] = np.matmul(self.a[-2].T, self.d[-1]) / X.shape[0]

        for i in range(len(self.neurons_per_layer)-1,0,-1):
            if self.use_improved_sigmoid == True:
                df = improved_sigmoid_derivative(self.z[i-1])
            else:
                df = self.a[i]*(1-self.a[i])

            temp = self.ws[i].dot(self.d[i].T)
            self.d[i-1] = (df.T*temp).T
            self.grads[i-1] = ((np.matmul( self.d[i-1].T, self.a[i-1])).T)/X.shape[0]



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
    X_train = pre_process_images(X_train, mean, std)
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
