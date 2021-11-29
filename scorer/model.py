import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

class SVCNN(nn.Module):
    """
    From https://github.com/jongchyisu/mvcnn_pytorch/blob/master/models/MVCNN.py
    """
    def __init__(self):
        super(SVCNN, self).__init__()

        self.net = EfficientNet.from_pretrained('efficientnet-b0')
        self.efficientnet = True

    def forward(self, x):
        output = self.net.extract_features(x)
        output = self.net._avg_pooling(output)
        return output

class MVCNN(nn.Module):
    """
    From https://github.com/jongchyisu/mvcnn_pytorch/blob/master/models/MVCNN.py
    """
    def __init__(self, num_views=1, num_classes=2):
        super(MVCNN, self).__init__()

        self.num_views = num_views

        self.net_1 = SVCNN()
        self.net_2 = nn.Linear(1280, num_classes)

    def forward(self, x):
        y = self.net_1(x)
        y = y.view((int(x.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1]))
        return F.log_softmax(self.net_2(torch.max(y,1)[0].view(y.shape[0],-1)))

if __name__ == "__main__":
    num_views = 6
    data = torch.zeros((32, num_views, 3, 224, 224))
    data = data.reshape(-1, 3, 224, 224)

    model = MVCNN(num_views=num_views)
    output = model(data)

    print(output.shape)
