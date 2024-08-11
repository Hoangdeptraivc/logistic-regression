import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ tập tin CSV
path_file_object = r'D:\data_set\logistic_regression\data_classification\data_classification.csv'
file_object = pd.read_csv(path_file_object)

# Tách dữ liệu thành đầu vào và đầu ra
x = file_object.iloc[:, :-1].to_numpy()
y = file_object.iloc[:, -1].to_numpy().reshape(-1, 1)

# Thêm cột một để xử lý phần bias
x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

# Hàm sigmoid
def sigmoid(s):
    return 1 / (1 + np.exp(-s))

# Hàm logistic regression
def logistic_regression(x, y, learning_rate=0.01, num_iterations=20000, tol=1e-6, check_w_after=100):
    w = np.random.randn(x.shape[1], 1)
    for i in range(num_iterations):
        z = np.dot(x, w)
        a = sigmoid(z)
        dw = np.dot(x.T, (a - y))
        w_new = w - learning_rate * dw

        if i % check_w_after == 0:
            if np.linalg.norm(w_new - w) < tol:
                print(f"Converged at iteration {i}")
                return w_new

        w = w_new

    return w

# Chạy logistic regression
weights = logistic_regression(x, y)
print("Final weights:\n", weights)

# Hàm vẽ đường phân loại
def plot_decision_boundary(x, y, weights):
    plt.figure(figsize=(10, 6))

    # Vẽ dữ liệu phân loại
    plt.scatter(x[y.flatten() == 0, 1], x[y.flatten() == 0, 2], color='red', label='Class 0')
    plt.scatter(x[y.flatten() == 1, 1], x[y.flatten() == 1, 2], color='blue', label='Class 1')

    # Vẽ đường phân loại
    x1_min, x1_max = x[:, 1].min(), x[:, 1].max()
    x2_min, x2_max = x[:, 2].min(), x[:, 2].max()
    x1_range = np.linspace(x1_min, x1_max, 100)
    x2_range = np.linspace(x2_min, x2_max, 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = sigmoid(weights[0] + weights[1] * X1 + weights[2] * X2)
    plt.contour(X1, X2, Z, levels=[0.5], colors='black', linewidths=1)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary')
    plt.legend()
    plt.show()

# Vẽ đồ thị
plot_decision_boundary(x, y, weights)
