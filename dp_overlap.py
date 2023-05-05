import time
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader


import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class DistDataset(object):
    # TODO: replace with the DataPartitioner stuff from the dist tutorial after this 
    # janky setup works.
    def __init__(self, data, train=True) -> None:
        self.data = data
        self.train = train
        self.n_splits = dist.get_world_size()
        # NOTE: assume this divides perfectly
        if train:
            self.chunk_size = len(self.data) // self.n_splits
        else:
            self.chunk_size = len(self.data)
    
    def get_rank_data(self):
        # does rank 0 need special treatment?
        # TODO: . Pass in a flag
        rank = dist.get_rank()
        if self.train:
            start_idx = self.chunk_size * rank
            end_idx = start_idx + self.chunk_size
        else:
            # Don't split test set. Just return the whole thing to whichever rank requests it.
            start_idx, end_idx = 0, self.chunk_size
        print(start_idx, end_idx)
        return [self.data[x] for x in range(start_idx, end_idx)]
        
        
def get_dist_data(is_train=True):
    transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
    # TODO: how to avoid rerunning code on all ranks that all needs to be run once. For
    # example, downloading dataset (this can be pinned to rank 0 and ask other ranks to wait
    # till the download is done - maybe some send/recv flag. How does PT handle it?) or 
    # using the same transform
    
    dataset = datasets.MNIST('../data', train=is_train, download=True,
                             transform=transform)
    
    dist_dataset = DistDataset(dataset, train=is_train).get_rank_data()
    dl = DataLoader(dist_dataset, batch_size=32)
    return dl


def avg_grads(model):
    world_size = dist.get_world_size()
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= world_size

    # avg = dist.all_reduce(model.parameters, torch.sum) / dist.get_world_size()
    # model.parameters = avg  

def train(model, dl, opt, device):
    rank = dist.get_rank()
    print(rank)
    for i, (x, y) in enumerate(dl):
        x, y = x.to(device), y.to(device)
        # print(i, rank)
        y_hat = model(x)
        loss = F.nll_loss(y_hat, y)
        # print(loss)
        loss.backward()
        # TODO: verify that doing the opt.step() first and avg the parameters is the same as
        # avg the grads and then doing the opt.step()
        
        # to mess up the all reduce 
        # if i == 0 and rank == 0:
        #     continue
        # avg_grads(model)
        opt.step()

# overlapping backward
# - for parameter in reversed order: (TODO: how to get layers from a module to run backward one at a time?)
# -- calculate backward
# -- trigger all-reduce
# - run opt.step 


def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def backward_allreduce_overlap_hook(grad):
    # print("I am a hook")
    world_size = dist.get_world_size()
    # async_op might be doing some async GD updates? Need to wait for all requests before calling .step
    dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=True)
    grad.data /= world_size
    
    
def register_hooks(model):
    # TODO: filter only requires_grad=True params?
    for p in model.named_parameters():
        p[1].register_hook(backward_allreduce_overlap_hook)
    
    
def run(rank, size, cuda):
    n_epochs = 2
    if cuda:
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")
    train_dl = get_dist_data(is_train=True)
    test_dl = get_dist_data(is_train=False)
    
    model = Net().to(device)
    register_hooks(model)
    
    opt = optim.SGD(model.parameters(), lr=1e-5)
    for i in range(n_epochs):
        print(f"starting training epoch {i}")
        train(model=model, dl=train_dl, opt=opt, device=device)
        # TODO: Messing up the number of all_reduce calls on one of the ranks should lead to a 
        # hanging process? Yes, yes it does - doe not hang but hits a timeout looks like? 
        # It does hang at the end of 1st epoch once the barrier calls are in. What is going on exactly?
        # TODO: check that adding these two barrier calls does nothing. 
        # dist.barrier()
        if rank == 0:
            test(model=model, test_loader=test_dl, device=device)
        # dist.barrier()
        

def init_process(rank, size, cuda, fn, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, cuda)
    

def main():
    size = 2
    cuda = True
    processes = []
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, cuda, run))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

# cpu, 1 = 34 
# cuda, 1 = 25 (gloo)
# cuda, 1 = 23 (nccl)
# cpu, 2 = 39
# cuda, 2 = 37 (gloo)
# cuda, 2 = 28 (nccl)
# cpu, 4 = 62 
# TODO: cuda, 4 = 60 (nccl) - WHY is the accuracy worse here (epoch 1 is 76 vs 86 and epoch 2 is 88 vs 90. rerun?)?

    
if __name__ == '__main__':
    t0 = time.time()
    main()
    print(time.time() - t0)
    

# TODO: dataset partitioner
# TODO: shuffle between epochs? Does that work in DP?
# TODO: Using SGD now. How would this work for Adam?
# TODO: add tensorboard. Does it work for dist?
# TODO: add cuda device. Use nccl backend? is it faster thabn gloo?
# TODO: time and see if there is an actual speed up. Do it for 1, 2, 4, 8 GPUs.
# TODO: DDP style bucketing for CC overlap. Autograd hooks? DDP hooks?
# TODO: implement all-reduce
            



# def train(args, model, device, train_loader, optimizer, epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))
#             if args.dry_run:
#                 break


# def test(model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)

#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))




# def main():
#     # Training settings
#     parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
#     parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#                         help='input batch size for training (default: 64)')
#     parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
#                         help='input batch size for testing (default: 1000)')
#     parser.add_argument('--epochs', type=int, default=14, metavar='N',
#                         help='number of epochs to train (default: 14)')
#     parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
#                         help='learning rate (default: 1.0)')
#     parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
#                         help='Learning rate step gamma (default: 0.7)')
#     parser.add_argument('--no-cuda', action='store_true', default=False,
#                         help='disables CUDA training')
#     parser.add_argument('--no-mps', action='store_true', default=False,
#                         help='disables macOS GPU training')
#     parser.add_argument('--dry-run', action='store_true', default=False,
#                         help='quickly check a single pass')
#     parser.add_argument('--seed', type=int, default=1, metavar='S',
#                         help='random seed (default: 1)')
#     parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                         help='how many batches to wait before logging training status')
#     parser.add_argument('--save-model', action='store_true', default=False,
#                         help='For Saving the current Model')
#     args = parser.parse_args()
#     use_cuda = not args.no_cuda and torch.cuda.is_available()
#     use_mps = not args.no_mps and torch.backends.mps.is_available()

#     torch.manual_seed(args.seed)

#     if use_cuda:
#         device = torch.device("cuda")
#     elif use_mps:
#         device = torch.device("mps")
#     else:
#         device = torch.device("cpu")

#     train_kwargs = {'batch_size': args.batch_size}
#     test_kwargs = {'batch_size': args.test_batch_size}
#     if use_cuda:
#         cuda_kwargs = {'num_workers': 1,
#                        'pin_memory': True,
#                        'shuffle': True}
#         train_kwargs.update(cuda_kwargs)
#         test_kwargs.update(cuda_kwargs)

#     transform=transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#         ])
#     dataset1 = datasets.MNIST('../data', train=True, download=True,
#                        transform=transform)
#     dataset2 = datasets.MNIST('../data', train=False,
#                        transform=transform)
#     train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
#     test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

#     model = Net().to(device)
#     optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

#     scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
#     for epoch in range(1, args.epochs + 1):
#         train(args, model, device, train_loader, optimizer, epoch)
#         test(model, device, test_loader)
#         scheduler.step()

#     if args.save_model:
#         torch.save(model.state_dict(), "mnist_cnn.pt")


