import  numpy as np

# ------------------------- 测试模块 -------------------------
def load_model(model_path, input_size, hidden1, hidden2, output_size):
    """加载预训练模型"""
    model = ThreeLayerNN(input_size, hidden1, hidden2, output_size)
    params = np.load(model_path)
    model.W1 = params['W1']
    model.b1 = params['b1']
    model.W2 = params['W2']
    model.b2 = params['b2']
    model.W3 = params['W3']
    model.b3 = params['b3']
    return model

def test_model(model, X_test, y_test):
    """测试模型性能"""
    probs = model.forward(X_test)
    preds = np.argmax(probs, axis=1)
    return np.mean(preds == y_test)