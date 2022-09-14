# coding: utf-8


class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y                
        out = x * y
        return out

    # 将上游传来的导数乘以正向传播的翻转值后传给下游
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out

    # 将上游传来的导数原封不动传给下游
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy
