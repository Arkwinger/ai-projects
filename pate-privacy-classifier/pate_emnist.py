import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from safetensors.torch import save_file

from torch.utils.data import TensorDataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------
# MODEL
# ------------------------------------------------

class MLP(nn.Module):

    def __init__(self, input_size=784, hidden_layers=None, num_classes=26, dropout=0.2):

        super(MLP, self).__init__()

        if hidden_layers is None:
            hidden_layers = [256, 128]

        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        prev_size = input_size

        for hidden_size in hidden_layers:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.dropouts.append(nn.Dropout(dropout))
            prev_size = hidden_size

        self.output = nn.Linear(prev_size, num_classes)

    def forward(self, x):

        for layer, dropout in zip(self.layers, self.dropouts):
            x = F.relu(layer(x))
            x = dropout(x)

        return self.output(x)

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

train_dataset = datasets.EMNIST(
    "data",
    split='letters',
    train=True,
    download=True
)

test_dataset = datasets.EMNIST(
    "data",
    split='letters',
    train=False,
    download=True
)

X_train = train_dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
X_test = test_dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0

y_train = train_dataset.targets.numpy() - 1
y_test = test_dataset.targets.numpy() - 1

# ------------------------------------------------
# NORMALIZATION
# ------------------------------------------------

scaler = StandardScaler()

X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# ------------------------------------------------
# SPLIT
# ------------------------------------------------

X_private, X_public, y_private, y_public = train_test_split(
    X_train_norm,
    y_train,
    test_size=0.3,
    random_state=42,
    stratify=y_train
)

print("Private:", len(X_private))
print("Public:", len(X_public))

# ------------------------------------------------
# HELPERS
# ------------------------------------------------

def create_loader(X, y, batch_size=128, shuffle=True):

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_model(model, loader, epochs=10):

    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):

        total_loss = 0

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

# ------------------------------------------------
# TRAIN TEACHERS
# ------------------------------------------------

NUM_TEACHERS = 10

indices = np.random.permutation(len(X_private))

partition_size = len(X_private) // NUM_TEACHERS

teachers = []

for i in range(NUM_TEACHERS):

    start = i * partition_size
    end = start + partition_size

    teacher_idx = indices[start:end]

    X_teacher = X_private[teacher_idx]
    y_teacher = y_private[teacher_idx]

    loader = create_loader(X_teacher, y_teacher)

    model = MLP()

    print(f"\nTraining teacher {i+1}/{NUM_TEACHERS}")

    train_model(model, loader, epochs=20)

    teachers.append(model)

print("\nAll teachers trained.")

# ------------------------------------------------
# VOTING
# ------------------------------------------------

def get_teacher_votes(teachers, X):

    votes = np.zeros((len(X), 26), dtype=np.int32)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    for teacher in teachers:

        teacher.eval()

        with torch.no_grad():

            outputs = teacher(X_tensor)

            preds = outputs.argmax(dim=1).cpu().numpy()

        for i, pred in enumerate(preds):
            votes[i, pred] += 1

    return votes

# ------------------------------------------------
# NOISY LABELS
# ------------------------------------------------

def noisy_labels(votes, noise_scale=1):

    noise = np.random.laplace(
        loc=0.0,
        scale=noise_scale,
        size=votes.shape
    )

    noisy_votes = votes + noise

    return np.argmax(noisy_votes, axis=1)

# ------------------------------------------------
# GENERATE STUDENT DATA
# ------------------------------------------------

votes = get_teacher_votes(teachers, X_public)

student_y = noisy_labels(votes, noise_scale=1)

X_student = X_public

print("\nStudent samples:", len(X_student))

# ------------------------------------------------
# TRAIN STUDENT
# ------------------------------------------------

student_loader = create_loader(
    X_student,
    student_y,
    batch_size=128
)

student_model = MLP()

print("\nTraining student model...")

train_model(student_model, student_loader, epochs=50)

# ------------------------------------------------
# SAVE
# ------------------------------------------------

save_file(
    student_model.state_dict(),
    "pate_student.safetensors"
)

print("\nSaved pate_student.safetensors")
