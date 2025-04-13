import os
import numpy as np
import pickle
import urllib.request
import tarfile


# ------------------------- 数据加载模块 -------------------------
def download_and_extract_cifar10(data_dir='./datasets/cifar-10'):
    """自动下载并解压CIFAR-10数据集"""
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    target_path = os.path.join(data_dir, "cifar-10-python.tar.gz")
    extracted_dir = os.path.join(data_dir, "cifar-10-batches-py")

    os.makedirs(data_dir, exist_ok=True)

    if os.path.exists(extracted_dir) and len(os.listdir(extracted_dir)) > 0:
        return extracted_dir

    if not os.path.exists(target_path):
        print("Downloading CIFAR-10 dataset...")
        urllib.request.urlretrieve(url, target_path)

    print("Extracting dataset...")
    with tarfile.open(target_path, 'r:gz') as tar:
        tar.extractall(path=data_dir)
    return extracted_dir

def load_cifar10(data_dir=None):
    """加载并预处理数据集"""
    if data_dir==None:
        data_dir='./datasets/cifar-10'

    cifar_dir = download_and_extract_cifar10(data_dir)

    def unpickle(file):
        with open(file, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
        return data

    # 加载训练集
    X_train, y_train = [], []
    for i in range(1, 6):
        data = unpickle(os.path.join(cifar_dir, f'data_batch_{i}'))
        X_train.append(data[b'data'])
        y_train.extend(data[b'labels'])
    X_train = np.vstack(X_train).astype(np.float32) / 255.0
    y_train = np.array(y_train)

    # 加载测试集
    test_data = unpickle(os.path.join(cifar_dir, 'test_batch'))
    X_test = test_data[b'data'].astype(np.float32) / 255.0
    y_test = np.array(test_data[b'labels'])

    return X_train, y_train, X_test, y_test