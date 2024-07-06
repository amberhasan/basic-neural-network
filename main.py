import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import gdown
import warnings
from sklearn.exceptions import ConvergenceWarning
# Install gdown to download files from Google Drive

# Suppress convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class NeuralNet:
    def __init__(self, dataFile):
        self.raw_input = pd.read_excel(dataFile)
        self.processed_data = None

    def preprocess(self):
        self.processed_data = self.raw_input
        self.processed_data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

        # Convert species names to integers
        label_encoder = LabelEncoder()
        self.processed_data['species'] = label_encoder.fit_transform(self.processed_data['species'])

        # Check if mapping is correct
        print(self.processed_data['species'].unique())

        # Check for missing values and handle them
        if self.processed_data.isnull().values.any():
            print("Data contains NaN values. Dropping or filling NaN values.")
            self.processed_data.dropna(inplace=True)  # Drop rows with NaN values

        # Standardize features
        X = self.processed_data.iloc[:, :-1]
        y = self.processed_data.iloc[:, -1]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        self.processed_data = pd.DataFrame(X_scaled, columns=self.processed_data.columns[:-1])
        self.processed_data['species'] = y

        # Check the processed data
        print(self.processed_data.head())
        print(self.processed_data['species'].isnull().sum())

        # Display all 150 rows
        pd.set_option('display.max_rows', 150)
        print(self.processed_data)

        return 0

    def train_evaluate(self):
        ncols = len(self.processed_data.columns)
        X = self.processed_data.iloc[:, 0:(ncols - 1)]
        y = self.processed_data.iloc[:, (ncols-1)]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        hidden_layer_sizes = [(5,), (10,), (5, 5)]
        learning_rates = ['constant', 'invscaling', 'adaptive']
        activations = ['logistic', 'tanh', 'relu']

        results = []
        histories = []

        for hl_size in hidden_layer_sizes:
            for lr in learning_rates:
                for activation in activations:
                    print(f'Training model with {hl_size}, {lr}, {activation}')
                    model = MLPClassifier(hidden_layer_sizes=hl_size, learning_rate=lr, activation=activation, max_iter=2000, random_state=42)
                    model.fit(X_train, y_train)

                    # Predict and evaluate
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)

                    train_accuracy = accuracy_score(y_train, y_train_pred)
                    test_accuracy = accuracy_score(y_test, y_test_pred)
                    train_mse = mean_squared_error(y_train, y_train_pred)
                    test_mse = mean_squared_error(y_test, y_test_pred)

                    # Record results
                    results.append({
                        'hidden_layer_sizes': hl_size,
                        'learning_rate': lr,
                        'activation': activation,
                        'train_accuracy': train_accuracy,
                        'test_accuracy': test_accuracy,
                        'train_mse': train_mse,
                        'test_mse': test_mse
                    })

                    # Save model history
                    histories.append((model.loss_curve_, f'{hl_size}, {lr}, {activation}'))

        # Determine the common y-axis range
        all_losses = [loss for history, _ in histories for loss in history]
        y_min, y_max = min(all_losses), max(all_losses)

        # Plot model histories in multiple subplots
        num_subplots = 3  # Number of subplots
        plots_per_subplot = len(histories) // num_subplots + (len(histories) % num_subplots > 0)

        fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 15), sharex=True)

        for i, ax in enumerate(axes):
            start = i * plots_per_subplot
            end = start + plots_per_subplot if i < num_subplots - 1 else len(histories)

            for history, label in histories[start:end]:
                ax.plot(history, label=label)

            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            ax.set_ylim(y_min, y_max)  # Set the same y-axis range for all plots
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True)

        plt.tight_layout()
        plt.show()

        results_df = pd.DataFrame(results)
        print(results_df)



# Download the file from Google Drive
file_id = '1cj3u1TinNh7PNbT1MM2mK-QuTqEgP34c'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'IrisData.xlsx'
gdown.download(url, output, quiet=False)

# Usage
data_file_path = output  # The downloaded file

nn = NeuralNet(data_file_path)
nn.preprocess()
nn.train_evaluate()


#%%
