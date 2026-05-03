import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model import DrugNet

print("Loading preprocessed data...")
X_train = np.load('X_train.npy')
X_test  = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test  = np.load('y_test.npy')

print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")

X_train_t = torch.FloatTensor(X_train)

y_train_t = torch.LongTensor(y_train)

X_test_t  = torch.FloatTensor(X_test)
y_test_t  = torch.LongTensor(y_test)

print("\nConverted to tensors -> Correct")

train_dataset = TensorDataset(X_train_t, y_train_t)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,

    shuffle=True
)

total_batches = len(train_loader)
print(f"Total batches per epoch: {total_batches}")

model     = DrugNet(input_dim=5000)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=5, gamma=0.5
)

EPOCHS = 10

train_losses    = []
train_accuracies = []

print("\n" + "="*50)
print("STARTING TRAINING")
print("="*50)

for epoch in range(EPOCHS):

    model.train()

    epoch_loss     = 0
    correct        = 0
    total          = 0

    for batch_num, (X_batch, y_batch) in enumerate(train_loader):

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

        predicted = torch.argmax(outputs, dim=1)

        correct += (predicted == y_batch).sum().item()

        total += y_batch.size(0)

        if (batch_num + 1) % 500 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_num+1}"
                  f"/{total_batches} | "
                  f"Loss: {loss.item():.4f}")
            
    avg_loss = epoch_loss / total_batches

    accuracy = (correct / total) * 100

    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)

    scheduler.step()

    print(f"\nEpoch {epoch+1}/{EPOCHS} Complete!")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Training Accuracy: {accuracy:.2f}%")
    print("-" * 40)

torch.save(model.state_dict(), 'drug_model.pth')

np.save('train_losses.npy',     np.array(train_losses))
np.save('train_accuracies.npy', np.array(train_accuracies))

print("\n" + "="*50)
print("TRAINING COMPLETE!")
print(f"   Final Loss:     {train_losses[-1]:.4f}")
print(f"   Final Accuracy: {train_accuracies[-1]:.2f}%")
print("   Model saved to: drug_model.pth")
print("="*50)