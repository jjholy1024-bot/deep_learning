import numpy as np
import pickle
from common.optimizer import Adam
from model import TwoLayerCNN

# Load dataset
file_path = 'cifar10_dataset.pkl'
with open(file_path, 'rb') as f:
    dataset = pickle.load(f)

# Assign datasets
x_train = dataset['x_train']
x_test = dataset['x_test']
t_train = dataset['t_train']
t_test = dataset['t_test']

# network
# conv_param 내의 변수값 지정 
# filter의 갯수는 각 convolution 층별로 10개와 15개
# filter크기는 3X3 크기
# pad는 0을 사용
# convolution filter의 stride는 1칸씩
# hidden_size 100, 
# output_size 10
network = TwoLayerCNN(input_dim=(3, 32, 32),
                      conv_param = {'filter_nums':(10, 15), 'filter_size': 3, 'pad': 0, 'stride':1},
                      hidden_size=100, output_size=10, weight_init_std=0.01)

# optimizer Adam 사용
optimizer = Adam() 

# 결과 저장용 list
train_loss_list=[]
train_acc_list=[]
test_acc_list=[]

# 학습중 횟수와 epoch을 세기 위한 변수 세팅
current_iter = 0
current_epoch = 0

num_train = 25000
mask = np.random.choice(x_train.shape[0], num_train, replace=False)
x_train = x_train[mask]
t_train = t_train[mask]

# iterations
# 20 에폭을 학습
# batch_size는 100으로
epochs = 20 
batch_size = 100 
train_size = x_train.shape[0]
iter_per_epoch = max(train_size / batch_size, 1)
max_iter = int(epochs * iter_per_epoch)

for i in range(max_iter):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    grads = network.backward(x_batch, t_batch)
    optimizer.update(network.params, grads)
    
    if current_iter % iter_per_epoch == 0:
        current_epoch += 1

        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"===epoch:{current_epoch}, train_acc:{train_acc}, test_acc:{test_acc}, loss:{loss}===")
    current_iter += 1

network.save_params('trained_TLC.pkl')