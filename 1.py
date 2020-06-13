class Perceptron(object):
    def __init__(self,input_num,activator):
        self.activator =activator
        self.weights = [0.0 for i in range(input_num)]
        self.bias = 0.0

    def __str__(self):
        return 'weigths\t:%s\nbias\t:%f\n' % (self.weights,self.bias)
    
    def predict(self, input_vec):
        return self.activator(
            reduce(lambda a,b: a + b,
                    map(lambda (x,w):x * w,
                        zip(input_vec,self.weights))
                ,0.0) + self.bias)

    def train(self, iniput_vecs,labels,iteration,rate)