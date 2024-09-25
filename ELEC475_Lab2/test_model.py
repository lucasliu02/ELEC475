import torch
from model import RegressionModel

def main():
    t = torch.rand([1, 3, 227, 227])
    print(t.shape)

    model = RegressionModel()
    t = model(t)
    print(t.shape)

if __name__ == '__main__':
    main()