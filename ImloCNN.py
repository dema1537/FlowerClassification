import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.transforms.v2 import ToTensor
import scipy
import matplotlib.pyplot as plt
import time

from torch.utils.data import DataLoader
from sklearn.metrics import precision_score


filename = "CNN.pth.tar"
device = "cpu"

# if torch.cuda.is_available():
#     print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
#     device = "cuda"

# else:
#     print("No GPU available. Training will run on CPU.")




testing = datasets.Flowers102(
    root = "ImageData",
    split = "test",
    download = True,
        transform = v2.Compose([

        v2.Resize((224,224), antialias=True),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
)


batchSize = 64

testingDataLoader = DataLoader(testing, batch_size=batchSize, shuffle=False)



Realtraining = datasets.Flowers102(
    root = "ImageData", 
    split = "train",
    download = True,
    transform = v2.Compose([

        v2.Resize((224,224), antialias=True),
        v2.RandomHorizontalFlip(),    
        v2.RandomVerticalFlip(),
        v2.RandomRotation(45),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        v2.RandomAdjustSharpness(sharpness_factor=2),
        v2.ElasticTransform(alpha=25.0),
        v2.RandomApply([v2.RandomErasing(p=1.0)], p=0.5),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ]),
)


Realvalidation = datasets.Flowers102(
    root = "ImageData",
    split = "val",
    download = True,
        transform = v2.Compose([
        v2.Resize((224,224), antialias=True),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ]),
)

RealtrainDataLoader = DataLoader(Realtraining, batch_size=batchSize, shuffle=True)
RealvalidationDataLoader = DataLoader(Realvalidation, batch_size=batchSize, shuffle=False)

# for x,y in RealtrainDataLoader:
#   x = x.to(device)
#   fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(12,8))
#   for i in range(2):
#     for j in range(4):
#       ax[i,j].imshow(x[i*4+j].cpu().permute(1,2,0))
#       ax[i,j].axis('off')
#   break

# plt.show()

class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(


            #1-16x224
            nn.Conv2d(3, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            #2
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),


            #3-32x112
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),


            #4
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),


            #5-64x56
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            

            #6
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),

            #7-128x28            
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            #8
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),

            #9-256x14
            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            #10
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),

            nn.Flatten(),

            
        )

        Nchannels = self.feature(torch.empty(1, 3, 224, 224)).size(-1)
        print(Nchannels)

        self.classify = nn.Sequential(



            #1
            nn.Linear(int(Nchannels), int(Nchannels * 2)),
            nn.ReLU(),
            nn.Dropout(0.2),


            #2
            nn.Linear(int(Nchannels * 2), int(Nchannels/8)),
            nn.ReLU(),
            nn.Dropout(0.5),


            #3
            nn.Linear(int(Nchannels/8), int(Nchannels/8)),
            nn.ReLU(),
            nn.Dropout(0.5),


            #4
            nn.Linear(int(Nchannels/8), int(102)),

        )

    def forward(self, x):
        features = self.feature(x)

        return self.classify(features)



classifier = CNN().to(device)

lossFunction = nn.CrossEntropyLoss()

optimiser = Adam(classifier.parameters(), lr=1e-4, weight_decay=1e-3)

state = {
                'epoch' : 0,
                'model' : classifier.state_dict(),
                'BestAcc' : 0,
            }
torch.save(state, filename)




def training(model, trainDataLoader, lossFunction, optimiser):

    model.train()

    for batch, (X,y) in enumerate(trainDataLoader):

        prediction = model(X)
        loss = lossFunction(prediction, y)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        loss, current = loss.item(), batch * batchSize + len(X)

        print(f"loss: {loss:>7f}  [{current:>5d}/{len(trainDataLoader.dataset):>5d}]")




def validating(model, testDataLoader, lossFunction):
    model.eval()

    size = len(testDataLoader.dataset)
    numberOfBatches = len(testDataLoader)

    testLoss = 0
    correct = 0

    with torch.no_grad():

        for X, y in testDataLoader:

            prediction = model(X)
            testLoss += lossFunction(prediction, y).item()
            correct += (prediction.argmax(1) == y).type(torch.float).sum().item()

        testLoss /= numberOfBatches
        correct /= size

        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {testLoss:>8f} \n")



        checkpoint = torch.load(filename)

        accu = (checkpoint['BestAcc'])

        

        if ((100 * correct) > accu) & ((100 * correct) > 45):
            accu = (100 * correct)
            print("Saving....")

            state = {
                'epoch' : t,
                'model' : classifier.state_dict(),
                'BestAcc' : accu,
            }

            torch.save(state, filename)

            checkpoint = torch.load(filename)

            print(checkpoint['BestAcc'])

epochs = 300

startTime = time.time()
runningTime = 11.5 * 60 * 60



for t in range(epochs):

    timeSoFar = time.time() - startTime

    if timeSoFar > runningTime:
        print("Time up, exiting training")
        break
    else:
        print(timeSoFar)

    print(f"Epoch {t+1}\n-------------------------------")

    training(model=classifier, trainDataLoader=RealtrainDataLoader, lossFunction=lossFunction, optimiser=optimiser)


    validating(classifier, RealvalidationDataLoader, lossFunction)

print("Done Training")

print("\n-\n-\n-\n-\n-\n-\n")

print("Commencing testing with testing data loader")


checkpoint = torch.load(f=filename, map_location=torch.device(device))
print(checkpoint['epoch'])
print(checkpoint['BestAcc'])


classifier.load_state_dict(checkpoint['model'])


def testing(model, testDataLoader, lossFunction):
    model.eval()

    size = len(testDataLoader.dataset)
    numberOfBatches = len(testDataLoader)

    testLoss = 0
    correct = 0

    with torch.no_grad():

        for X, y in testDataLoader:

            prediction = model(X)
            testLoss += lossFunction(prediction, y).item()
            correct += (prediction.argmax(1) == y).type(torch.float).sum().item()

        testLoss /= numberOfBatches
        correct /= size

        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {testLoss:>8f} \n")



testing(classifier, testingDataLoader, lossFunction)

print(".......Done.......")
