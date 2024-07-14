import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import load_iris


class NeuralNet:
    def __init__(self):
        # Load Iris dataset
        iris = load_iris()
        self.raw_input = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                                      columns=iris['feature_names'] + ['target'])
        self.label_encoder = LabelEncoder()
        # Encode target labels with value between 0 and n_classes-1.
        self.raw_input['target'] = self.label_encoder.fit_transform(self.raw_input['target'])

    def preprocess(self):
        scaler = StandardScaler()
        features = self.raw_input.drop('target', axis=1)
        features_scaled = scaler.fit_transform(features)
        self.processed_data = pd.DataFrame(features_scaled, columns=features.columns)
        self.processed_data['target'] = self.raw_input['target']
        return 0

    def train_evaluate(self):
        X = self.processed_data.drop('target', axis=1).values
        y = self.processed_data['target'].values
        y = tf.keras.utils.to_categorical(y, num_classes=3)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Hyperparameters
        activations = ['sigmoid', 'relu', 'tanh']
        optimizers = ['adam', 'sgd']
        losses = ['categorical_crossentropy', 'mean_squared_error']
        epochs = [50, 200]
        num_hidden_layers_options = [4, 10]

        results = []

        for activation in activations:
            for optimizer_name in optimizers:
                for loss in losses:
                    for epoch in epochs:
                        for num_layers in num_hidden_layers_options:
                            model = Sequential()
                            model.add(Input(shape=(X_train.shape[1],)))

                            # Let's add hidden layers
                            for _ in range(num_layers):
                                model.add(Dense(8, activation=activation))

                            model.add(Dense(3, activation='softmax'))

                            # Get the optimizer
                            if optimizer_name == 'adam':
                                optimizer = tf.keras.optimizers.Adam()
                            else:
                                optimizer = tf.keras.optimizers.SGD()

                            model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
                            history = model.fit(X_train, y_train, epochs=epoch, batch_size=10, validation_data=(X_test, y_test), verbose=0)

                            train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
                            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

                            results.append({
                                'Activation': activation,
                                'Optimizer': optimizer_name,
                                'Loss Function': loss,
                                'Epochs': epoch,
                                'Number of Layers': num_layers,
                                'Training Accuracy (%)': round(train_acc * 100, 2),
                                'Testing Accuracy (%)': round(test_acc * 100, 2),
                                'Training Loss (%)': round(train_loss * 100, 2),
                                'Testing Loss (%)': round(test_loss * 100, 2)
                            })

                            plt.figure()
                            plt.plot(history.history['accuracy'], label='Train Accuracy')
                            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
                            plt.title(f'Model Accuracy: {activation}-{optimizer_name}-{loss}-{epoch} epochs')
                            plt.ylabel('Accuracy (%)')
                            plt.xlabel('Epoch')
                            plt.legend(loc='upper left')
                            plt.savefig(f"{activation}_{optimizer_name}_{loss}_{epoch}_layers_{num_layers}.png")
                            plt.close()

        results_df = pd.DataFrame(results)
        results_df.to_csv('model_results.csv', index=False)
        print(results_df)


if __name__ == "__main__":
    neural_net = NeuralNet()
    neural_net.preprocess()
    neural_net.train_evaluate()

#%%
