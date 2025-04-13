
import  numpy as np
from model.model import ThreeLayerNN

# ------------------------- 训练模块 -------------------------
def train_model(model, X_train, y_train, X_val, y_val, epochs=50,
                batch_size=128, lr=0.001, reg_lambda=0.0001, lr_decay=0.95):
    """训练过程"""
    best_val_acc = 0.0
    best_params = None
    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(epochs):
        lr *= lr_decay  # 学习率衰减
        indices = np.random.permutation(len(X_train))

        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i + batch_size]
            X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]

            # 前向传播
            probs = model.forward(X_batch)

            # 计算损失
            log_probs = -np.log(np.clip(probs[range(len(y_batch)), y_batch], 1e-10, 1.0))
            data_loss = np.mean(log_probs)
            reg_loss = 0.5 * reg_lambda * (np.sum(model.W1 ** 2) + np.sum(model.W2 ** 2) + np.sum(model.W3 ** 2)) / len(
                y_batch)
            total_loss = data_loss + reg_loss
            train_losses.append(total_loss)

            # 反向传播
            dW1, db1, dW2, db2, dW3, db3 = model.backward(X_batch, y_batch, reg_lambda)

            # 参数更新
            model.W1 -= lr * dW1
            model.b1 -= lr * db1
            model.W2 -= lr * dW2
            model.b2 -= lr * db2
            model.W3 -= lr * dW3
            model.b3 -= lr * db3

        # 验证集评估
        val_probs = model.forward(X_val)
        val_pred = np.argmax(val_probs, axis=1)
        val_acc = np.mean(val_pred == y_val)
        val_accs.append(val_acc)

        val_loss = -np.log(val_probs[range(len(y_val)), y_val]).mean() + \
                   0.5 * reg_lambda * (np.sum(model.W1 ** 2) + np.sum(model.W2 ** 2) + np.sum(model.W3 ** 2)) / len(
            y_val)
        val_losses.append(val_loss)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = {
                'W1': model.W1.copy(),
                'b1': model.b1.copy(),
                'W2': model.W2.copy(),
                'b2': model.b2.copy(),
                'W3': model.W3.copy(),
                'b3': model.b3.copy()
            }
            np.savez('best_model.npz', **best_params)

        print(f"Epoch {epoch + 1}/{epochs} | Val Acc: {val_acc:.4f} | Loss: {val_loss:.4f}")

    # 恢复最佳参数
    model.W1, model.b1 = best_params['W1'], best_params['b1']
    model.W2, model.b2 = best_params['W2'], best_params['b2']
    model.W3, model.b3 = best_params['W3'], best_params['b3']

    return train_losses, val_losses, val_accs

# ------------------------- 超参数搜索模块 -------------------------
def hyperparameter_search(X_train, y_train, X_val, y_val):
    """超参数网格搜索"""
    hidden_sizes = [(128, 64), (256, 128)]  # (hidden1, hidden2)
    learning_rates = [0.01, 0.001]
    reg_lambdas = [0.001, 0.0001]
    results = []

    for hs in hidden_sizes:
        for lr in learning_rates:
            for reg in reg_lambdas:
                print(f"\nTraining with hs={hs}, lr={lr}, reg={reg}")
                model = ThreeLayerNN(3072, hs[0], hs[1], 10, 'relu')
                train_loss, val_loss, val_acc = train_model(
                    model, X_train, y_train, X_val, y_val,
                    epochs=20,  # 快速搜索使用较少epochs
                    batch_size=128,
                    lr=lr,
                    reg_lambda=reg,
                    lr_decay=1.0  # 关闭学习率衰减
                )
                results.append({
                    'hidden_size': hs,
                    'lr': lr,
                    'reg': reg,
                    'val_acc': max(val_acc)
                })

    best = max(results, key=lambda x: x['val_acc'])
    print("\nBest Parameters:")
    print(f"Hidden Sizes: {best['hidden_size']}")
    print(f"Learning Rate: {best['lr']}")
    print(f"Regularization: {best['reg']}")
    print(f"Validation Accuracy: {best['val_acc']:.4f}")
    return best