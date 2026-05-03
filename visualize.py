import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import confusion_matrix
import seaborn as sns

train_losses     = np.load('train_losses.npy')
train_accuracies = np.load('train_accuracies.npy')
test_predictions = np.load('test_predictions.npy')
y_test           = np.load('y_test.npy')

epochs = range(1, len(train_losses) + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('DrugNet — Training Performance', fontsize=15, fontweight='bold', y=1.01)

ax1 = axes[0]
ax1.plot(epochs, train_losses, color='#e74c3c', linewidth=2.5,
         marker='o', markersize=6, label='Training Loss')
ax1.fill_between(epochs, train_losses, alpha=0.1, color='#e74c3c')
ax1.set_title('Loss Over Epochs', fontsize=13, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax1.legend(fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.set_facecolor('#f9f9f9')

ax2 = axes[1]
ax2.plot(epochs, train_accuracies, color='#2ecc71', linewidth=2.5,
         marker='s', markersize=6, label='Training Accuracy')
ax2.fill_between(epochs, train_accuracies, alpha=0.1, color='#2ecc71')

TEST_ACCURACY = 92.69
ax2.axhline(y=TEST_ACCURACY, color='#3498db', linewidth=2,
            linestyle='--', label=f'Test Accuracy ({TEST_ACCURACY}%)')

ax2.set_title('Accuracy Over Epochs', fontsize=13, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Accuracy (%)', fontsize=11)
ax2.set_ylim(75, 101)
ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax2.legend(fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.set_facecolor('#f9f9f9')

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
print("✅ Saved: training_curves.png")
plt.show()

cm     = confusion_matrix(y_test, test_predictions)
labels = ['Bad Drug (0)', 'Good Drug (1)']

fig2, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=labels,
    yticklabels=labels,
    linewidths=0.5,
    linecolor='gray',
    annot_kws={"size": 13, "weight": "bold"},
    ax=ax
)

ax.set_title('Confusion Matrix — Test Set', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Actual Label',    fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)

fp_val = cm[0][1]
ax.text(1.5, -0.35,
        f'⚠  {fp_val:,} bad drugs predicted as good (False Positives)',
        ha='center', fontsize=10, color='#c0392b',
        transform=ax.get_xaxis_transform())

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✅ Saved: confusion_matrix.png")
plt.show()

print("\nDone! Both images saved to your project folder.")