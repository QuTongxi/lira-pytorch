import os
import time
import torch
import copy
import torch.ao.quantization as tq
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
class PTQBasicBlock(models.resnet.BasicBlock):
    def __init__(self, block): 
        self.__dict__ = dict(vars(block))
        self.add_relu = nn.quantized.FloatFunctional()
        
    def convert(origin):
        return PTQBasicBlock(origin)    
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add_relu.add_relu(out, identity)
        return out
    
    def fuse_model(self) -> None:
        tq.fuse_modules(self, ["conv1", "bn1", "relu"],inplace=True)
        tq.fuse_modules(self, ["conv2", "bn2"],inplace=True)
        if self.downsample:
            tq.fuse_modules(self.downsample, ["0", "1"], inplace=True)


class MyPTQModel(models.ResNet):
    def __init__(self, model):
        self.__dict__ = dict(vars(model))        
                
        self.quant = tq.QuantStub() 
        self.dequant = tq.DeQuantStub()
        
        self.eval()
        tq.fuse_modules(self, ["conv1", "bn1", "relu"], inplace=True)
        
        arr = []
        for child in self.modules():
            if type(child) == models.resnet.BasicBlock:
                ch = PTQBasicBlock.convert(child)
                ch.fuse_model()   
                arr.append(ch)  
        
        self.layer1 = nn.Sequential(arr[0],arr[1])
        self.layer2 = nn.Sequential(arr[2],arr[3])
        self.layer3 = nn.Sequential(arr[4],arr[5])
        self.layer4 = nn.Sequential(arr[6],arr[7])                               
                                
                       
    def forward(self, x):
        x = self.quant(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        x = self.dequant(x)
        return x               

        
    def origin_size(self):
        print_size_of_model(self)
        return
    
    def print_child(self):
        for nm,ch in self.named_children():
            print(nm, ch)
                   


def runPTQ(model, calibrate_dl, test_dl):
    tmp = copy.deepcopy(model)
    m = MyPTQModel(tmp)
    m.cpu()         
    m.origin_size()
    m.qconfig = tq.QConfig(
    activation=tq.MinMaxObserver.with_args(dtype=torch.quint8),
    weight=tq.MinMaxObserver.with_args(dtype=torch.qint8)  # 设置权重为 qint8
)
    tq.prepare(m, inplace=True) 
    
    calibrate_model(m, calibrate_dl)
    # get_acc(m, test_dl)
    print(f"[test in quant] acc_test: {get_acc(m, test_dl):.4f}")
    
    tq.convert(m, inplace=True)    
    print_size_of_model(m)
    # print(f"[test] acc_test: {get_acc(m, test_dl):.4f}")
    return    

        

def runPTQ_1(model, calibrate_dl, test_dl):
    nums = [1024]
    for num in nums:
        run_step(num,model,calibrate_dl,test_dl)


def run_step(argv,model, calibrate_dl, test_dl):
    tmp = copy.deepcopy(model)
    m = MyPTQModel(tmp)
    m.cpu()         
    m.origin_size()
    m.qconfig = tq.QConfig(
    activation=tq.HistogramObserver.with_args(
        dtype=torch.quint8, 
        qscheme=torch.per_tensor_affine,
        bins = argv
    ),
    weight=tq.PerChannelMinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_channel_affine
    )
)

    tq.prepare(m, inplace=True) 
    
    calibrate_model(m, calibrate_dl)
    print(f"[test] acc_test: {get_acc(m, test_dl):.4f}")
    
    tq.convert(m, inplace=True)    
    print_size_of_model(m)
    print(f"[test] acc_test: {get_acc(m, test_dl):.4f}")
    return







def calibrate_model(model, data_loader):
    """Run the calibration step to collect statistics for quantization."""
    model.eval()
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to("cpu")
            model(x)


@torch.no_grad()
def get_acc(model, dl):
    model.cpu()
    start_time = time.time()  # 记录开始时间
    
    acc = []
    for x, y in dl:
        x, y = x.to("cpu"), y.to("cpu")
        acc.append(torch.argmax(model(x), dim=1) == y)
    
    s = len(acc)
    acc = torch.cat(acc)
    acc = torch.sum(acc) / len(acc)

    end_time = time.time()  # 记录结束时间
    elapsed_time = (end_time - start_time) / s    # 计算运行时间
    print(f"Time taken for get_acc: {elapsed_time:.4f} seconds")  # 打印运行时间

    return acc.item()




def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')