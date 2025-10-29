"""
Script to evaluate a trained GFNet model on the ADNI test dataset and visualize predictions.
"""

import torch
from torch.utils.data import DataLoader
from dataset import get_adni_dataloader, ADNIDataset, ADNI_ROOT_PATH, TEST_TRANSFORM 
from modules import GFNet
import random
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns


def evaluate_model(model, device, test_loader):
    """
    Evaluates the model on the test dataset and prints accuracy.
    Args:
        model: Trained model to evaluate.
        device (torch.device): Torch device (CPU or GPU) for computation.
        test_loader (DataLoader): DataLoader for the test dataset.
    """
    model.eval()  
    correct = 0
    total = 0

    with torch.no_grad(): 
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}%')

def plot_confusion_matrix(model, device, test_loader):
    """
    Plots confusion matrix and classification report.
    Args:
        model: Trained model to evaluate.
        device (torch.device): Torch device (CPU or GPU) for computation.
        test_loader (DataLoader): DataLoader for the test dataset.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    classes = ['NC', 'AD']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    return cm


def plot_roc_curve(model, device, test_loader):
    """
    Plots ROC curve and calculates AUC score.
    Args:
        model: Trained model to evaluate.
        device (torch.device): Torch device (CPU or GPU) for computation.
        test_loader (DataLoader): DataLoader for the test dataset.
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            
            all_probs.extend(probabilities[:, 1].cpu().numpy())  # Probabilities for positive class
            all_labels.extend(labels.cpu().numpy())
    
    # calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"AUC Score: {roc_auc:.3f}")
    return roc_auc

def do_predictions(model, device: torch.device, num_predictions: int = 9, show_plot: bool = True):
    """
    Makes predictions on random samples from the test dataset and visualizes them.
    Args:
        model: Trained model to use for predictions.
        device (torch.device): Torch device (CPU or GPU) for computation.
        num_predictions (int): Number of random samples to predict and visualize.
        show_plot (bool): Whether to display the plot interactively.
    """

    test_dataset = ADNIDataset(ADNI_ROOT_PATH, train=False, transform=TEST_TRANSFORM)
    nrc = math.ceil(math.sqrt(num_predictions))
    fig, axes = plt.subplots(nrows=nrc, ncols=nrc, squeeze=False, figsize=(math.ceil(224 * nrc / 100), math.ceil(224 * nrc / 100)))

    for i in range(num_predictions):
        # Randomly select samples
        idx = random.randint(0, len(test_dataset) - 1)
        image, label = test_dataset[idx]
        image_tensor = image.unsqueeze(0).to(device)  
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        pred = int(predicted[0].item())
        label_strs = {0: "NC", 1: "AD"} # Label mappings

        image = image.permute(1, 2, 0).cpu().numpy() # Prepare image for display
        ax = axes[i // nrc, i % nrc]
        ax.imshow(image)
        ax.set_title(f"Pred: {label_strs[pred]}, True: {label_strs[label]}")
        ax.axis('off')

    for i in range(num_predictions, nrc * nrc):
        ax = axes[i // nrc, i % nrc]
        ax.axis('off')

    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.savefig("predictions.png")
    if show_plot:
        plt.show()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = GFNet(depth=8).to(device)
    # Load the trained model weights
    model.load_state_dict(torch.load('gfnet_model2.pth'))
    print("Model loaded successfully.")

    batch_size = 32  
    # Get test DataLoader
    test_loader = get_adni_dataloader(batch_size=batch_size, train=False) 

    evaluate_model(model, device, test_loader)

    plot_confusion_matrix(model, device, test_loader)

    plot_roc_curve(model, device, test_loader)
    # Make and visualize predictions
    do_predictions(model, device, num_predictions=9, show_plot=True) 

if __name__ == "__main__":
    main()
