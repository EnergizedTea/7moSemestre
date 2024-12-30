'''
Mi intento de perceptron

x = [1,2,3,4,5,6]
w = [6,-5,4,3,2,1]
y = []
yx = []
for i in range(len(x)):
    y.append(x[i] * w[i])
print(y)
    
for i in y:
   if i > 1:
      # True
      yx.append(1)
   else:
      # False
      yx.append(0)
print(yx)
'''
import random

class Perceptron():
    def __init__(self, learning_rate = 0.1, epochs=1000):
        self.bias = 1
        self.weight = []
        self.learning_rate = learning_rate
        self.epochs = epochs
        

    def suma_ponderada(self, data):
        #print('entraste')
        x = 0
        for i in range(len(data)):
            x = x + (self.weight[i]*data[i])
        return x

    def funcion_activada(self, suma):
        return 1 if suma >= 0 else 0
    
    def predecir(self,data):
        return 1 if self.funcion_activada(self.suma_ponderada(data) + self.bias) == 1 else 0
    
    def calculo_error(self, predict, real):
        error = (real - predict)
        return error
        
    
    def train(self, x, y):
        self.weight = [random.randint(-1,1) for i in range(len(x))]
        for i in range(self.epochs):
            pred = (self.predecir(x))
            error = self.calculo_error(pred, y)
            self.bias += (self.learning_rate * error)
            print(self.bias)
            for j in range(len(x)):
                self.weight[j] = self.weight[j] + (error*self.learning_rate*x[j])
        '''print('---------')
        print(self.weight)
        print('---------')'''
        print(pred)


# Si la moneda es una X, sera 1, si es un circulo sera 0

x = [
    1,0,0,0,1,
    0,1,0,0,1,
    0,0,1,0,0,
    0,1,0,1,0,
    1,0,0,0,1
]
o = [
    0,0,1,0,0,
    0,1,0,1,0,
    1,0,0,0,1,
    0,1,0,1,0,
    0,0,1,0,0
]

moneda = [
    0,0,1,0,0,
    0,1,0,1,0,
    1,0,0,0,1,
    0,1,0,1,0,
    0,0,1,0,0
]

p = Perceptron()
p.train(x, 1)
print(p.predecir(moneda))


