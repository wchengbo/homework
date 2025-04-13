from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from train_and_test.train import hyperparameter_search,train_model
import numpy as np
from model.data import load_cifar10
from model.model import ThreeLayerNN
from train_and_test.test import  test_model
# ------------------------- 可视化模块 -------------------------
def plot_training_curves(train_loss, val_loss, val_acc):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(np.linspace(0, len(train_loss), len(val_loss)), val_loss, label='Val Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_acc)
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()


def visualize_weights(model):
    """可视化第一层权重"""
    W1 = model.W1
    plt.figure(figsize=(10, 5), facecolor='white')
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow((W1[:, i].reshape(32, 32, 3) - W1.min()) / (W1.max() - W1.min()),
                   vmin=0, vmax=1)
        plt.axis('off')
    plt.savefig('weight_visualization.png')
    plt.show()


if __name__ == "__main__":
    # 加载数据
    X_train, y_train, X_test, y_test = load_cifar10('./datasets/cifar-10')
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # 超参数搜索
    # best_params = hyperparameter_search(X_train, y_train, X_val, y_val)

    # 使用最佳参数训练最终模型
    final_model = ThreeLayerNN(
        input_size=3072,
        hidden1=256,
        hidden2=128,
        output_size=10,
        activation='relu'
    )

    train_loss, val_loss, val_acc = train_model(
        final_model,
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=128,
        lr=0.01,
        reg_lambda=0.001,
    )

    # 测试最终模型
    test_acc = test_model(final_model, X_test, y_test)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    # 可视化
    plot_training_curves(train_loss, val_loss, val_acc)
    visualize_weights(final_model)