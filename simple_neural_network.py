import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Input Variables :
# - Number of times pregnant
# - Plasma glucose concentration at 2 hours in an oral glucose tolerance test
# - Diastolic blood pressure (mm Hg)
# - Triceps skin fold thickness (mm)
# - 2-hour serum insulin (μIU/ml)
# - Body mass index (weight in kg/(height in m)2)
# - Diabetes pedigree function
# - Age (years)
#
# Output: 1 for diabetes or 0 for not diabetes

# Load the dataset, split into trainingset and testset
def load_dataset_and_transform_it():
    dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
    X_ = dataset[:,0:8]
    y_ = dataset[:,8]

    X_train, X_test, y_train, y_test = train_test_split( X_, y_, test_size=0.33, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    return X_train, y_train, X_test, y_test


# Create a pytorch y_test
def create_pytorch_model():
    model = nn.Sequential(
        nn.Linear(8, 12),
        nn.ReLU(),
        nn.Linear(12, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid()
    )
    print(model)
    return model


# Create a more verbose model
def create_custom_pytorch_model():
    class PimaClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden1 = nn.Linear(8, 12)
            self.act1 = nn.ReLU()
            self.hidden2 = nn.Linear(12, 8)
            self.act2 = nn.ReLU()
            self.output = nn.Linear(8, 1)
            self.act_output = nn.Sigmoid()

        def forward(self, x):
            x = self.act1(self.hidden1(x))
            x = self.act2(self.hidden2(x))
            x = self.act_output(self.output(x))
            return x

    return PimaClassifier()


def training(model, X, y):
    n_epochs = 100
    batch_size = 10

    for epoch in range(n_epochs):
        for i in range(0, len(X), batch_size):
            Xbatch = X[i:i + batch_size]
            y_pred = model(Xbatch)
            ybatch = y[i:i + batch_size]
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Finished epoch {epoch}, latest loss {loss}')

    print("Training Done.")


if __name__ == "__main__":

    # Step 1 - Load dataset
    X, y, X_test, y_test = load_dataset_and_transform_it()

    # Step 2 - Create NN model
    model1 = create_pytorch_model()
    #model1 = create_pytorch_model()

    # Step 3 - Define the loss function
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model1.parameters(), lr=0.001)

    # Step 4 - Start training
    training(model=model1, X=X, y=y)

    # Step 5 - Prediction
    # make probability predictions using the testset
    predictions = (model1(X_test) > 0.5).int()
    # Compute accuracy on prediction
    accuracy = float((predictions == y_test).float().mean())
    print(f"Accuracy {accuracy}")
