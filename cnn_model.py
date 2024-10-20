import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load dataset 
custom_dataset = ImageFolder(root='data/train_data', transform=transform)

# Split the dataset into training and test sets
train_size = int(0.8 * len(custom_dataset))
test_size = len(custom_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [train_size, test_size])

# Create DataLoader for training and test sets
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)

# Define the CNN model
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 512)  # Changed from 28 * 28 to 16 * 16
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 16 * 16)  # Changed from 28 * 28 to 16 * 16
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
class_names = custom_dataset.classes
num_classes = len(class_names)
model = CNN(num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

print("Training finished!")

# Save the model
torch.save(model.state_dict(), 'cnn_model.pth')

# Function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Show images
imshow(torchvision.utils.make_grid(images))


# model.load_state_dict(torch.load('cnn_model.pth'))
# model.eval()

# outputs = model(images.to(device))
# _, predicted = torch.max(outputs, 1)

# print('Predicted: ', ' '.join(f'{custom_dataset.classes[predicted[j]]:5s}' for j in range(len(outputs))))

# Load the saved model and make predictions on test data
def evaluate_model(model, test_loader, criterion, device):
    model.load_state_dict(torch.load('cnn_model.pth'))
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient computation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy

test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")

def predict_image(model, image_path, transform, device, class_names):
    model.load_state_dict(torch.load('cnn_model.pth'))
    model.eval()
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    return class_names[predicted.item()]

# Example usage
image_path = 'data/test_data/strawberry_2.jpg'
predicted_class = predict_image(model, image_path, transform, device, class_names)
print(f"Predicted class: {predicted_class}")

# def visualize_predictions(model, test_loader, class_names, num_images=8):
#     model.load_state_dict(torch.load('cnn_model.pth'))
#     model.eval()
#     images_so_far = 0
#     fig = plt.figure(figsize=(10, 10))
    
#     with torch.no_grad():
#         for i, (inputs, labels) in enumerate(test_loader):
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
            
#             for j in range(inputs.size()[0]):
#                 images_so_far += 1
#                 ax = plt.subplot(num_images // 2, 2, images_so_far)
#                 ax.axis('off')
#                 ax.set_title(f'predicted: {class_names[preds[j]]}')
#                 imshow(inputs.cpu().data[j])
                
#                 if images_so_far == num_images:
#                     return
#     plt.tight_layout()
#     plt.show()
def visualize_predictions(model, test_loader, class_names, num_images=5):
    model.load_state_dict(torch.load('cnn_model.pth'))
    model.eval()
    
    # Get a batch of images
    images, labels = next(iter(test_loader))
    images = images.to(device)
    labels = labels.to(device)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    # Determine the grid size
    num_images = min(num_images, len(images))
    rows = int(np.ceil(np.sqrt(num_images)))
    cols = int(np.ceil(num_images / rows))
    
    # Create a figure with a grid of subplots
    fig = plt.figure(figsize=(4*cols, 4*rows))
    
    for i in range(num_images):
        # Display the image
        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(images[i].cpu().permute(1, 2, 0))
        ax.axis('off')
        ax.set_title(f'True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}')
    
    plt.tight_layout()
    plt.show()
    
    # Display prediction probabilities
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    axes = axes.flatten() if num_images > 1 else [axes]
    
    for i in range(num_images):
        ax = axes[i]
        probs = torch.nn.functional.softmax(outputs[i], dim=0)
        ax.bar(range(len(class_names)), probs.cpu())
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90)
        ax.set_ylim([0, 1])
        ax.set_title(f'Image {i+1} Probabilities')
    
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

# Call this function after training
visualize_predictions(model, test_loader, class_names, num_images=15) 


def visualize_predictions(model, test_loader, class_names, num_images=5):
    model.load_state_dict(torch.load('cnn_model.pth'))
    model.eval()
    
    # Get a batch of images
    images, labels = next(iter(test_loader))
    images = images.to(device)
    labels = labels.to(device)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    # Determine the grid size
    num_images = min(num_images, len(images))
    rows = num_images
    cols = 2  # One column for images, one for probability graphs
    
    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    fig.suptitle('Predictions with Probability Distributions', fontsize=16)
    
    for i in range(num_images):
        # Display the image
        ax_img = axes[i, 0]
        ax_img.imshow(images[i].cpu().permute(1, 2, 0))
        ax_img.axis('off')
        ax_img.set_title(f'True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}')
        
        # Display the prediction probabilities
        ax_prob = axes[i, 1]
        probs = torch.nn.functional.softmax(outputs[i], dim=0)
        ax_prob.bar(range(len(class_names)), probs.cpu())
        ax_prob.set_xticks(range(len(class_names)))
        ax_prob.set_xticklabels(class_names, rotation=0, ha='right')
        ax_prob.set_ylim([0, 1])
        ax_prob.set_title('Class Probabilities')
        ax_prob.set_ylabel('Probability')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.5)  # Adjust to prevent title overlap
    plt.show()

# Call this function after training
visualize_predictions(model, test_loader, class_names, num_images=10) 