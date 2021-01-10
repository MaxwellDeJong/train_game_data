# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 11:39:25 2020

@author: Max
"""

from eval_net import load_model, load_one_hot_dict
from load_resnet import load_resnet50
import numpy as np
import matplotlib.pyplot as plt
from flashtorch.saliency import Backprop
from flashtorch.utils import denormalize, format_for_plotting, standardize_and_clip
from torchvision import transforms
from network_utils import load_normalization_stats
from torch.nn import Softmax

def custom_plot(backprop, input_, target_class, guided=False, figsize=(3, 12), use_gpu=False, cmap='viridis', alpha=0.5):
    gradients = backprop.calculate_gradients(input_,
                                     target_class,
                                     guided=guided,
                                     use_gpu=use_gpu)
    
    max_gradients = backprop.calculate_gradients(input_,
                                             target_class,
                                             guided=guided,
                                             take_max=True,
                                             use_gpu=use_gpu)

    # Setup subplots
    subplots = [
        # (title, [(image1, cmap, alpha), (image2, cmap, alpha)])
        ('Input image',
         [(format_for_plotting(denormalize(input_)), None, None)]),
        ('Gradients across RGB channels',
         [(format_for_plotting(standardize_and_clip(gradients)),
          None,
          None)]),
        ('Max gradients',
         [(format_for_plotting(standardize_and_clip(max_gradients)),
          cmap,
          None)]),
        ('Overlay',
         [(format_for_plotting(denormalize(input_)), None, None),
          (format_for_plotting(standardize_and_clip(max_gradients)),
           cmap,
           alpha)])
    ]

    fig = plt.figure(figsize=figsize)

    for i, (title, images) in enumerate(subplots):
        ax = fig.add_subplot(len(subplots), 1, i + 1)
        ax.set_axis_off()
        ax.set_title(title)

        for image, cmap, alpha in images:
            ax.imshow(image, cmap=cmap, alpha=alpha)
                

plt.close('all')

model = load_resnet50(use_custom=True, dirname='ski-race2')
load_model(model, dirname='ski-race2')
print('Model loaded')
backprop = Backprop(model)

test_img = np.load('D:/steep_training/ski-race2/balanced/training_frame-18231.npy')
print('Image shape: ', test_img.shape)
plt.imshow(test_img)
plt.axis('off')
plt.show()
print('Image loaded')
(means, stds) = load_normalization_stats(dirname='ski-race2')

print('loaded means: ', means)
print('loaded stds: ', stds)
frame_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
       ])
    
input_ = frame_transform(test_img).unsqueeze(0)
input_.requires_grad = True

m = Softmax(dim=1)
pred = m(model(input_))
print('prediction: ', pred)
print(load_one_hot_dict(dirname='ski-race2'))

print('Image transformed')
#target_class = [1, 0, 0]
target_class = 1
#grads = backprop.calculate_gradients(test_img, )

(grads, max_gradients) = backprop.visualize(input_, target_class, guided=False, return_output=True, figsize=(16, 3))
plt.show()

custom_plot(backprop, input_, target_class)
plt.show()

plt.figure(frameon=False)
grads = max_gradients.numpy()[0, :, :]
plt.imshow(grads, alpha=0.9, cmap=plt.cm.brg, vmin=np.min(grads), vmax=np.max(grads))
plt.imshow(test_img, alpha=0.5)
plt.axis('off')
plt.show()