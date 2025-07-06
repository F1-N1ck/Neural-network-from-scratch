import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from neural_network import NeuralNetwork

# Load MNIST
mnist = fetch_openml('mnist_784', version=1, as_frame=True)
X = mnist['data']
y = mnist['target'].astype(np.int32).to_numpy().reshape(-1, 1)

X = X / 255.0

encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

X_train_df, X_test_df, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

X_train = X_train_df.to_numpy()
X_test = X_test_df.to_numpy()

# Initialize and train
nn = NeuralNetwork(input_size=784, hidden_size1=128, hidden_size2=64, output_size=10, learning_rate=0.1, decay_rate=0.95)

nn.train(X_train, y_train, epochs=10, batch_size=64)

# Evaluate
preds = nn.predict(X_test)
true = np.argmax(y_test, axis=1)
acc = np.mean(preds == true)
print(f"Test accuracy: {acc*100:.2f}%")
