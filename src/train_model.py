import pandas as pd
import numpy as np
import joblib

class MyStandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit_transform(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return (X - self.mean) / (self.std + 1e-8)

    def transform(self, X):
        return (X - self.mean) / (self.std + 1e-8)

class MyLabelEncoder:
    def __init__(self):
        self.classes = {}
        self.inverse_classes = {}

    def fit_transform(self, y):
        unique_labels = np.unique(y)
        for i, label in enumerate(unique_labels):
            self.classes[label] = i
            self.inverse_classes[i] = label
        return np.array([self.classes[label] for label in y])

    def inverse_transform(self, y_indices):
        return [self.inverse_classes[i] for i in y_indices]

class MyLinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

class MyLogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

class MyNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        self.lr = learning_rate

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _sigmoid_derivative(self, z):
        return z * (1 - z)

    def fit(self, X, y, iterations=2000):
        y_one_hot = np.zeros((y.size, y.max() + 1))
        y_one_hot[np.arange(y.size), y] = 1

        for i in range(iterations):
            z1 = np.dot(X, self.W1) + self.b1
            a1 = self._sigmoid(z1)
            z2 = np.dot(a1, self.W2) + self.b2
            output = self._sigmoid(z2)

            error = output - y_one_hot
            d_output = error * self._sigmoid_derivative(output)
            error_hidden = d_output.dot(self.W2.T)
            d_hidden = error_hidden * self._sigmoid_derivative(a1)

            self.W2 -= self.lr * a1.T.dot(d_output)
            self.b2 -= self.lr * np.sum(d_output, axis=0, keepdims=True)
            self.W1 -= self.lr * X.T.dot(d_hidden)
            self.b1 -= self.lr * np.sum(d_hidden, axis=0, keepdims=True)

    def predict(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self._sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        output = self._sigmoid(z2)
        return np.argmax(output, axis=1)

def train_manual_models():
    df = pd.read_csv('../data/delhi_aqi_data.csv')
    
    X = df[['punjab_fire_count', 'wind_speed_kmph', 'wind_dir_deg', 'temp_min_c']].values
    y_value = df['delhi_aqi'].values
    y_class = (df['delhi_aqi'] > 400).astype(int).values
    y_reason_raw = df['dominant_reason'].values

    scaler = MyStandardScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = MyLabelEncoder()
    y_reason = encoder.fit_transform(y_reason_raw)

    lin_reg = MyLinearRegression(learning_rate=0.001, iterations=5000)
    lin_reg.fit(X_scaled, y_value)

    log_reg = MyLogisticRegression(learning_rate=0.01, iterations=5000)
    log_reg.fit(X_scaled, y_class)

    input_size = X_scaled.shape[1]
    hidden_size = 8
    output_size = len(np.unique(y_reason))
    
    nn_model = MyNeuralNetwork(input_size, hidden_size, output_size)
    nn_model.fit(X_scaled, y_reason)

    models = {
        'linear': lin_reg,
        'logistic': log_reg,
        'neural_net': nn_model,
        'scaler': scaler,
        'encoder': encoder
    }
    joblib.dump(models, 'all_models_manual.pkl')

if __name__ == "__main__":
    train_manual_models()
