import numpy as np
from common.layers import SoftmaxWithLoss, Relu, Affine, Convolution, Pooling
from collections import OrderedDict
import pickle

class TwoLayerCNN:
    def __init__(self, input_dim=(3, 32, 32), 
                 conv_param = {'filter_nums':(10, 15), 
                               'filter_size':3, 
                               'pad':0,
                               'stride':1},
                 hidden_size=50, output_size=10, weight_init_std=0.01):
        # conv_param 내의 변수값 지정 
        filter_num1 = conv_param['filter_nums'][0] 
        filter_num2 = conv_param['filter_nums'][1]
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        # 계층간 출력맵 크기 결정
        # 주의점 : 입력맵과 출력맵의 크기 계산을 정확하게 하기 위해서 몫을 계산하는 (//)를 사용
        # ... 부분 작성
        input_size = input_dim[1]
        conv_output_size1 = (input_size + 2*filter_pad-filter_size) // filter_stride + 1
        pool_output_size1 = conv_output_size1 // 2
        conv_output_size2 = (pool_output_size1 + 2*filter_pad - filter_size) // filter_stride + 1
        pool_output_size2 = conv_output_size2 // 2

        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std*np.random.randn(filter_num1, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num1)
        self.params['W2'] = weight_init_std*np.random.randn(filter_num2, filter_num1, filter_size, filter_size)
        self.params['b2'] = np.zeros(filter_num2)
        self.params['W3'] = weight_init_std*np.random.randn(filter_num2 * pool_output_size2 * pool_output_size2, hidden_size)
        self.params['b3'] = np.zeros(hidden_size)
        self.params['W4'] = weight_init_std*np.random.randn(hidden_size, output_size)
        self.params['b4'] = np.zeros(output_size)
                
        # 계층 생성 아래의 구조에 맞게 생성
        # Conv1 -> Relu -> Pool -> Conv2 -> Relu -> Pool 
        # -> Hidden(Affine) -> Relu -> Hidden(Affine) -> Softmax -> output
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'], conv_param['stride'], conv_param['pad'])
        self.layers['Relu2'] = Relu()
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['Relu3'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W4'], self.params['b4'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        """
        순전파 연산
        입력: x (배치 크기, 채널, 높이, 너비)의 4차원 배열
        출력: 예측된 점수 (배치 크기, 클래스 수)
        """
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
        
    def loss(self, x, t):
        """
        손실 함수 (Cross Entropy 등) 계산
        입력: x(이미지 데이터), t(정답 레이블)
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)
                
    def backward(self, x, t):
        """
        역전파 연산 (gradient 계산)
        입력: x(이미지 데이터), t(정답 레이블)
        출력: 가중치에 대한 기울기를 담은 딕셔너리 (self.params와 동일한 키 사용)
        """
        self.loss(x, t)
        dout = self.last_layer.backward(1)
        
        layers = list(self.layers.values())
        layers.reverse()
        
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Conv2'].dW
        grads['b2'] = self.layers['Conv2'].db
        grads['W3'] = self.layers['Affine1'].dW
        grads['b3'] = self.layers['Affine1'].db
        grads['W4'] = self.layers['Affine2'].dW
        grads['b4'] = self.layers['Affine2'].db

        return grads
        
    def accuracy(self, x, t):
        """
        test data와 train data에 대한 학습 정확도 계산
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y==t)/float(x.shape[0])
        return accuracy

    def save_params(self, file_name="best_model.pkl"):
        """학습된 파라미터를 pickle로 저장"""
        with open(file_name, 'wb') as f:
            pickle.dump(self.params, f)
            
    def load_params(self, file_name="best_model.pkl"):
        """pickle 파일에서 파라미터를 불러옴"""
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key][:] = val[:] 