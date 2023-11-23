import random
import math
import copy
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import messagebox
import plotly.express as px
import torch
import torch.nn as nn
import torch.optim as optim

class KMeansAlgorithm():
    # ... (rest of your code)

    def visualize_clusters(self):
        # Convert the dataset and classes to a DataFrame
        df = pd.DataFrame(self.dataset, columns=[f'Dim_{i}' for i in range(self.numDim)])
        df['Cluster'] = [self.classes[i].index(1) for i in range(len(self.classes))]

        # Visualize the clusters using seaborn
        sns.pairplot(df, hue='Cluster', palette='viridis', markers=["o", "s", "D"])
        plt.show()

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

def train_linear_regression_model(data, input_dim):
    x = torch.tensor(data[:, :-1], dtype=torch.float32)
    y = torch.tensor(data[:, -1], dtype=torch.float32)

    model = LinearRegressionModel(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    num_epochs = 1000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    return model

# Tkinter GUI
def run_kmeans():
    kmeans = KMeansAlgorithm(data, K=int(entry_k.get()), threshold=float(entry_threshold.get()))
    kmeans.fit()
    kmeans.visualize_clusters()

def run_linear_regression():
    input_dim = len(data[0]) - 1
    model = train_linear_regression_model(data, input_dim)
    messagebox.showinfo("Linear Regression Result", f"Trained Linear Regression Model:\n{model}")

# Example data
random.seed(42)
data = [[random.uniform(0, 10) for _ in range(2)] for _ in range(300)]
kmeans = KMeansAlgorithm(data, K=3, threshold=0.1)
kmeans.fit()

# Tkinter GUI setup
root = tk.Tk()
root.title("KMeans and Linear Regression")

label_k = tk.Label(root, text="K Value:")
label_threshold = tk.Label(root, text="Threshold Value:")

entry_k = tk.Entry(root)
entry_threshold = tk.Entry(root)

button_kmeans = tk.Button(root, text="Run KMeans", command=run_kmeans)
button_linear_regression = tk.Button(root, text="Run Linear Regression", command=run_linear_regression)

label_k.grid(row=0, column=0)
label_threshold.grid(row=1, column=0)

entry_k.grid(row=0, column=1)
entry_threshold.grid(row=1, column=1)

button_kmeans.grid(row=2, column=0, columnspan=2, pady=10)
button_linear_regression.grid(row=3, column=0, columnspan=2, pady=10)

root.mainloop()
