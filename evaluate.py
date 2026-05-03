import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from model import DrugNet

print("Loading test data...")
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

test_dataset = TensorDataset(X_test_t, y_test_t)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model = DrugNet(input_dim=5000)
model.load_state_dict(torch.load('drug_model.pth', map_location=device, weights_only=True))
model.to(device)

model.eval()

print("\nRunning predictions on test set...")

all_preds  = []
all_labels = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)

        outputs   = model(X_batch)
        predicted = torch.argmax(outputs, dim=1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_batch.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

test_accuracy = accuracy_score(all_labels, all_preds) * 100

print("\n" + "="*50)
print("TEST SET RESULTS")
print("="*50)
print(f"  Test Accuracy:     {test_accuracy:.2f}%")

try:
    train_accs = np.load('train_accuracies.npy')
    train_acc  = train_accs[-1]
    gap        = train_acc - test_accuracy
    print(f"  Train Accuracy:    {train_acc:.2f}%")
    print(f"  Accuracy Gap:      {gap:.2f}%")

    print()
    if gap < 2:
        print("VERDICT: Great generalization — no significant overfitting!")
    elif gap < 5:
        print("VERDICT: Mild overfitting — model is slightly memorizing.")
    else:
        print("VERDICT: Overfitting detected — model memorized training data.")
        print("     Consider: more dropout, weight decay, or fewer epochs.")
except FileNotFoundError:
    pass

print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
print(classification_report(
    all_labels,
    all_preds,
    target_names=["Bad Drug (0)", "Good Drug (1)"]
))

print("="*50)
print("CONFUSION MATRIX")
print("="*50)
cm = confusion_matrix(all_labels, all_preds)
print(f"\n                 Predicted Bad  Predicted Good")
print(f"  Actual Bad      {cm[0][0]:>10}      {cm[0][1]:>10}")
print(f"  Actual Good     {cm[1][0]:>10}      {cm[1][1]:>10}")

tn, fp, fn, tp = cm.ravel()
print(f"\n  True Negatives  (correctly caught bad drugs):  {tn:,}")
print(f"  True Positives  (correctly found good drugs):  {tp:,}")
print(f"  False Positives (bad drug called good — RISKY): {fp:,}")
print(f"  False Negatives (good drug called bad):        {fn:,}")

np.save('test_predictions.npy', all_preds)
print("\nPredictions saved to test_predictions.npy")
print("="*50)