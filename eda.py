import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from collections import Counter
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Set the path to your dataset
data_dir = 'data/original/train_data'

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the dataset
dataset = ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 1. Visualize sample images
def show_sample_images(dataset, num_samples=10):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        idx = np.random.randint(len(dataset))
        img, label = dataset[idx]
        axes[i].imshow(img.permute(1, 2, 0))
        axes[i].set_title(f"Class: {dataset.classes[label]}")
        axes[i].axis('off')
    plt.show()

show_sample_images(dataset)


# 2. Analyze image dimensions and aspect ratios
def analyze_image_dimensions(data_dir):
    widths, heights, aspect_ratios = [], [], []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                with Image.open(img_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
                    aspect_ratios.append(w / h)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.hist(widths, bins=20)
    ax1.set_title('Width Distribution')
    ax2.hist(heights, bins=20)
    ax2.set_title('Height Distribution')
    ax3.hist(aspect_ratios, bins=20)
    ax3.set_title('Aspect Ratio Distribution')
    plt.tight_layout()
    plt.show()

analyze_image_dimensions(data_dir)


# 3. Examine class distribution
class_counts = Counter(dataset.targets)
plt.figure(figsize=(10, 5))
plt.bar(dataset.classes, [class_counts[i] for i in range(len(dataset.classes))])
plt.title('Class Distribution')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 4. Analyze pixel value distributions
def analyze_pixel_distributions(dataloader):
    channel_means = []
    channel_stds = []
    for images, _ in dataloader:
        channel_means.append(images.mean(dim=[0, 2, 3]))
        channel_stds.append(images.std(dim=[0, 2, 3]))
    
    channel_means = torch.stack(channel_means).mean(dim=0)
    channel_stds = torch.stack(channel_stds).mean(dim=0)
    
    print(f"Channel means: {channel_means}")
    print(f"Channel stds: {channel_stds}")
    
    plt.figure(figsize=(10, 5))
    for i, color in enumerate(['red', 'green', 'blue']):
        sns.kdeplot(images[:, i, :, :].flatten().numpy(), shade=True, color=color)
    plt.title('Pixel Value Distributions')
    plt.xlabel('Pixel Value')
    plt.ylabel('Density')
    plt.show()

analyze_pixel_distributions(dataloader)

# 5. Check for corrupted images
def check_corrupted_images(data_dir):
    corrupted = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except:
                    corrupted.append(img_path)
    
    if corrupted:
        print("Corrupted images found:")
        for path in corrupted:
            print(path)
    else:
        print("No corrupted images found.")

check_corrupted_images(data_dir)

# 6. Visualize average images per class
def show_average_images(dataset):
    class_images = {c: [] for c in dataset.classes}
    for img, label in dataset:
        class_images[dataset.classes[label]].append(img.numpy())
    
    fig, axes = plt.subplots(1, len(dataset.classes), figsize=(20, 4))
    for i, (class_name, images) in enumerate(class_images.items()):
        avg_image = np.mean(images, axis=0)
        axes[i].imshow(avg_image.transpose(1, 2, 0))
        axes[i].set_title(class_name)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

show_average_images(dataset)

def extract_features(model, dataloader):
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for images, batch_labels in dataloader:
            batch_features = model(images).view(images.shape[0], -1)
            features.append(batch_features)
            labels.append(batch_labels)
    return torch.cat(features), torch.cat(labels)

def dimensionality_reduction(features, labels, method='pca', n_components=2):
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")
    
    reduced_features = reducer.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=labels, palette='deep')
    plt.title(f'{method.upper()} of Image Features')
    plt.show()


# For feature extraction and dimensionality reduction, we'll use a pre-trained model
print("Extracting features and performing dimensionality reduction...")
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
features, labels = extract_features(model, dataloader)

print("PCA visualization...")
dimensionality_reduction(features, labels, method='pca')

print("t-SNE visualization...")
dimensionality_reduction(features, labels, method='tsne')