import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
from keras.preprocessing.image import array_to_img, img_to_array
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import rotate


def load_data():
    
    #set image width and image height
    img_dim = 256
    
    images = []
    labels = []
    cat2label = defaultdict(lambda: None)
    label2cat = defaultdict(lambda: None)
    next_label = 0
    for root, dirs, files in os.walk("AI4ALL/project_images"):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".JPG"):
                img_file = os.path.join(root, file)
                img = Image.open(img_file)
                img.thumbnail((img_dim, img_dim))
                images.append(img_to_array(img))
                
                img_category = root.split("/")[2]
                if cat2label[img_category] is None:
                    cat2label[img_category] = next_label
                    label2cat[next_label] = img_category
                    next_label = next_label + 1
                label = cat2label[img_category]
                labels.append(label)
    images = np.asarray(images) / 255
    labels = np.asarray(labels)
    return images, labels, label2cat

def split_train_test(images, labels):
    indices = np.random.permutation(images.shape[0])
    
    split_point = int(np.ceil(images.shape[0] * 0.75))
    training_idx = indices[:split_point]
    testing_idx = indices[split_point:]
    
    train_images = images[training_idx]
    train_labels = labels[training_idx]
    
    test_images = images[testing_idx]
    test_labels = labels[testing_idx]
    
    return train_images, train_labels, test_images, test_labels
    
    

def augment_data(images, labels):
    final_images = []
    final_labels = []
    
    for idx in range(images.shape[0]):
        image = images[idx]
        label = labels[idx]
        options = [0, 1, 2, 3, 4]
        version = np.random.choice(options)
        
        final_images.append(image)
        final_labels.append(label)
        
        
        if version == 1:
            #flip image vertically
            flipped_image = np.flip(image, axis=0)
            final_images.append(flipped_image)
            final_labels.append(label)
        elif version == 2:
            #flip image horizontally
            flipped_image = np.flip(image, axis=1)
            final_images.append(flipped_image)
            final_labels.append(label)
        elif version == 3:
            #blur image
            blurred_image = gaussian_filter(image, sigma=0.5)
            final_images.append(blurred_image)
            final_labels.append(label)
        elif version == 4:
            #rotate image
            angle = np.random.rand() * 30
            rotated_image = rotate(image, angle, mode='nearest')
            final_images.append(rotated_image)
            final_labels.append(label)
    
    final_images = np.asarray(final_images)
    final_labels = np.asarray(final_labels)
    return final_images, final_labels

def display_images(images):
    np.random.shuffle(images)
    f, axarr = plt.subplots(2,2)
    axarr[0, 0].axis("off")
    axarr[0, 1].axis("off")
    axarr[1, 0].axis("off")
    axarr[1, 1].axis("off")
    axarr[0,0].imshow(images[0])
    axarr[0,1].imshow(images[1])
    axarr[1,0].imshow(images[2])
    axarr[1,1].imshow(images[3])
    
def display_class_counts(labels):
    classes = np.arange(labels.min(), labels.max()+1)
    counts = np.zeros(labels.max()-labels.min()+1)
    for label in labels.tolist():
        counts[label] = counts[label] + 1
    plt.bar(classes, counts)
    plt.xlabel("class")
    plt.ylabel("frequency")
