import  numpy as np
class ThreeLayerNN:
    """三层神经网络分类器"""

    def __init__(self, input_size, hidden1, hidden2, output_size, activation='relu'):
        # He初始化
        self.W1 = np.random.randn(input_size, hidden1) * np.sqrt(2. / input_size)
        self.b1 = np.zeros(hidden1)
        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2. / hidden1)
        self.b2 = np.zeros(hidden2)
        self.W3 = np.random.randn(hidden2, output_size) * np.sqrt(2. / hidden2)
        self.b3 = np.zeros(output_size)
        self.activation = activation

    def forward(self, X):
        """前向传播"""
        self.Z1 = X.dot(self.W1) + self.b1
        if self.activation == 'relu':
            self.A1 = np.maximum(0, self.Z1)
        elif self.activation == 'sigmoid':
            self.A1 = 1 / (1 + np.exp(-self.Z1))

        self.Z2 = self.A1.dot(self.W2) + self.b2
        if self.activation == 'relu':
            self.A2 = np.maximum(0, self.Z2)
        elif self.activation == 'sigmoid':
            self.A2 = 1 / (1 + np.exp(-self.Z2))

        self.Z3 = self.A2.dot(self.W3) + self.b3
        exp_scores = np.exp(self.Z3 - np.max(self.Z3, axis=1, keepdims=True))
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def backward(self, X, y, reg_lambda):
        """反向传播"""
        num_samples = X.shape[0]

        # 输出层梯度
        delta3 = self.probs.copy()
        delta3[range(num_samples), y] -= 1
        delta3 /= num_samples

        dW3 = self.A2.T.dot(delta3) + (reg_lambda / num_samples) * self.W3
        db3 = np.sum(delta3, axis=0)

        # 第二隐藏层梯度
        delta2 = delta3.dot(self.W3.T)
        if self.activation == 'relu':
            delta2 *= (self.Z2 > 0)
        elif self.activation == 'sigmoid':
            delta2 *= self.A2 * (1 - self.A2)

        dW2 = self.A1.T.dot(delta2) + (reg_lambda / num_samples) * self.W2
        db2 = np.sum(delta2, axis=0)

        # 第一隐藏层梯度
        delta1 = delta2.dot(self.W2.T)
        if self.activation == 'relu':
            delta1 *= (self.Z1 > 0)
        elif self.activation == 'sigmoid':
            delta1 *= self.A1 * (1 - self.A1)

        dW1 = X.T.dot(delta1) + (reg_lambda / num_samples) * self.W1
        db1 = np.sum(delta1, axis=0)

        return dW1, db1, dW2, db2, dW3, db3