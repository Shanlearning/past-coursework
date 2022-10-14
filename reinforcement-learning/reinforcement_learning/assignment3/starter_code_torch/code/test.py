import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

###############################################################################
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

###############################################################################
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
###############################################################################
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
###############################################################################
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

train(train_dataloader, model, loss_fn, optimizer)

dataloader = train_dataloader
size = len(dataloader.dataset)
model.train()
for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)

    # Compute prediction error
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch % 100 == 0:
        loss, current = loss.item(), batch * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

###############################################################################



def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
###############################################################################
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


#network = build_mlp( input_size = 5, output_size = 6, n_layers = 1, size = 64)
action_dim =6
policy = GaussianPolicy(network,action_dim)
policy.parameters()


import torch

tensor = torch.tensor([[1, 0, 2, 3, 4],[1, 0, 2, 3, 4]])
tensor.shape # torch.Size([5])
tensor.unsqueeze(dim=0).shape # [1, 5]
tensor.unsqueeze(dim=1).shape # [5, 1]

tensor.flatten().ndim

torch.squeeze(tensor).shape

    
tensor.squeeze().shape







        #actions = distribution.sample()
        actions.shape
        
        distribution_sum = ptd.Normal( torch.mean(distribution.mean,dim=1),torch.mean(distribution.covariance_matrix,dim = [1,2]) )
        
        loss = torch.mean(distribution_sum.log_prob(torch.mean(y,dim= [1,2]))) 
        
        
        
        distribution.stddev.shape
        distribution.covariance_matrix.shape
        
        ptd.Normal(, )
        
        loss = distribution.log_prob(actions) * torch.mean( torch.square(actions - y.flatten(start_dim=1)), dim = 1) 
        #torch.mean( distribution.log_prob(actions) * torch.mean( torch.square(actions - y.flatten(start_dim=1)), dim = 1) )
            
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}]")
            all_loss.append(loss)
            plt.plot(all_loss)
            plt.show()
            with open('results\\' + str(env.year) + '.json', 'w') as fp:
                json.dump(all_loss, fp)  





def test(env,policy,optimizer):
    policy.eval()
    X, y = sample_from_environment(env,training = True)
    distribution = policy.action_distribution(X.float())     
    distribution.mean
    distribution.stddev
    y.flatten(start_dim=1)




