
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
import numpy as np
image = Image.open("images/zebra.jpg")
print("Image shape:", image.size)

model = torchvision.models.resnet18(pretrained=True)
print(model)
print(type(model))
first_conv_layer = model.conv1
print("First conv layer weight shape:", first_conv_layer.weight.shape)
print("First conv layer:", first_conv_layer)

# Resize, and normalize the image with the mean and standard deviation
image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = image_transform(image)[None]
print("Image shape:", image.shape, type(image))

activation = first_conv_layer(image)
print("Activation shape:", activation.shape)


def torch_image_to_numpy(image: torch.Tensor):
    """
    Function to transform a pytorch tensor to numpy image
    Args:
        image: shape=[3, height, width]
    Returns:
        iamge: shape=[height, width, 3] in the range [0, 1]
    """
    # Normalize to [0 - 1.0]
    image = image.detach().cpu() # Transform image to CPU memory (if on GPU VRAM)
    image = image - image.min()
    image = image / image.max()
    image = image.numpy()
    if len(image.shape) == 2: # Grayscale image, can just return
        return image
    assert image.shape[0] == 3, "Expected color channel to be on first axis. Got: {}".format(image.shape)
    image = np.moveaxis(image, 0, 2)
    return image


def plot_weight_image(indices, activation, name):
    if (len(indices) % 2 == 1):
        subplot_size_y = 1
        subplot_size_x = len(indices)
    else:
        subplot_size_y = 2
        subplot_size_x = len(indices)//2
    #Plot weight image
    fig, axs = plt.subplots(subplot_size_y,subplot_size_x, figsize=(15, 6), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.001)
    axs = axs.ravel()

    for i in range(len(indices)):
        index = indices[i]
        img = activation.detach().numpy()[0][index]
        axs[i].imshow(img, cmap="ocean")
        axs[i].set_title(str(index))

    plt.tight_layout()

    plt.savefig(name +".png")
    plt.show()

def find_activation(model: torchvision.models.resnet.ResNet, num_filters_to_skip: int, image):

    num_filters = len(list(model.children()))
    stop = num_filters - num_filters_to_skip
    i = 0
    for m in model.children():
        if (i == stop):
            return image
        activ = m
        image = activ(image)
        i += 1

## 4b

activation_4b = activation
indices_4b = [14, 26, 32, 49, 52]
plot_weight_image(indices_4b, activation_4b, "task4c")

## 4c

activation_4c = find_activation(model, 2, image)
print("activation shape: ",activation.shape)
indices_4c = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

plot_weight_image(indices_4c, activation_4c, "task4c")