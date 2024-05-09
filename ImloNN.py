import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.transforms.v2 import ToTensor
import scipy




from torch.utils.data import Subset

trainingData = datasets.Flowers102(
    root = "ImageData",
    split = "train",
    download = True,
    transform = v2.Compose([
    
        v2.Resize((224,224), antialias=True),
        #v2.RandomHorizontalFlip(),
        v2.ToTensor(),
        #v2.Normalize(torch.Tensor(mean), torch.Tensor(std))

    ]),
)


testing = datasets.Flowers102(
    root = "ImageData",
    split = "test",
    download = True,
        transform = v2.Compose([

        v2.Resize((224,224), antialias=True),
        v2.ToTensor(),
        #v2.Normalize(torch.Tensor(mean), torch.Tensor(std)),
        
    ]),
)

validation = datasets.Flowers102(
    root = "ImageData",
    split = "val",
    download = True,
        transform = v2.Compose([
        v2.ToTensor(),
        v2.Resize((224,224), antialias=True),
        
    ]),
)

# print(testing)
# print(testing)
# print(validation)

batchSize = 64
dropOut = 0.5

trainDataLoader = DataLoader(trainingData, batch_size=batchSize, shuffle=True)
testDataLoader = DataLoader(testing, batch_size=batchSize)
validationDataLoader = DataLoader(validation, batch_size=batchSize)

def getMeanAndSTD():
    mean = 0
    std = 0

    totalImages = 0

    for images, _ in trainDataLoader:
        imageBatchCount = images.size(0)
        images = images.view(imageBatchCount, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        totalImages += imageBatchCount

    mean /= totalImages
    std /= totalImages

    return mean, std

meanCalc, stdCalc = getMeanAndSTD()

mean = [meanCalc,meanCalc,meanCalc]

std = [stdCalc,stdCalc,stdCalc]

Realtraining = datasets.Flowers102(
    root = "ImageData",
    split = "train",
    download = True,
    transform = v2.Compose([
    
        v2.RandomResizedCrop((224,224), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(10),
        v2.ToTensor(),
        #v2.RandomPerspective(0.3, 0.5),
        #v2.Normalize(torch.Tensor(meanCalc), torch.Tensor(stdCalc))

    ]),
)

RealtrainDataLoader = DataLoader(Realtraining, batch_size=batchSize, shuffle=True)


class CNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.feature = nn.Sequential(

            
            # nn.Conv2d(3, 16, kernel_size=3, stride=1),
            # nn.ReLU(),

            # nn.MaxPool2d(kernel_size=3, stride=2),
            # nn.ReLU(),

            # nn.Conv2d(16, 32, kernel_size=3, stride=1),
            # nn.ReLU(),

            # nn.Conv2d(32, 32, kernel_size=3, stride=1),
            # nn.ReLU(),

            # nn.MaxPool2d(kernel_size=3, stride=2),
            # nn.ReLU(),

            # nn.Conv2d(32, 64, kernel_size=3, stride=1),
            # nn.ReLU(),

            # nn.MaxPool2d(kernel_size=3, stride=3),
            # nn.ReLU(),

            nn.Conv2d(3, 16, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3, stride=4),
            nn.ReLU(),



            # nn.Conv2d(32, 64, 3),
            # nn.ReLU(),

            # # nn.Conv2d(256, 512, 3),
            # # nn.ReLU(),

            # # nn.MaxPool2d(kernel_size=3, stride=2),
            # # nn.ReLU(),

            # # nn.Conv2d(512, 512, 3),
            # # nn.ReLU(),

            # nn.MaxPool2d(kernel_size=3, stride=2),
            # nn.ReLU(),

            # nn.MaxPool2d(kernel_size=3, stride=2),
            # nn.ReLU(),


            nn.Flatten(),

        )

        Nchannels = self.feature(torch.empty(1, 3, 224, 224)).size(-1)

        self.classify = nn.Sequential(

    
            nn.Linear(int(Nchannels), int(Nchannels/128)),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(int(Nchannels/128), int(102)),
            
            
           

        )

        
        

    def forward(self, x):
        features = self.feature(x)
        
        return self.classify(features)

        
    
classifier = CNN().to("cpu")

optimiser = Adam(classifier.parameters(), lr=(batchSize/32)*0.001, betas=(0.9, 0.999))

lossFunction = nn.CrossEntropyLoss()


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

        return (100*correct)

        


def Savecheckpoint(classifier, epoch, optimiser, bestAccuraccy):
    print("Saving....")

    state = {
        'epoch' : epoch,
        'model' : classifier.state_dict(),
        'BestAcc' : bestAccuraccy,
        'optimiser' : optimiser.state_dict()
    }

    torch.save(state, 'SimpleModel10epochs.pth.tar')

def Loadcheckpoint(checkpoint):
    print("Loading...")

    classifier.load_state_dict(checkpoint['stateDict'])
    #optimiser.load(checkpoint['optimiser'])


# loadModel = False

# if loadModel == True:
#     Loadcheckpoint(torch.load("SimpleModelTrans.pth.tar"))

#     testing(classifier, testDataLoader, lossFunction)

epochs = 60

# for t in range(epochs):

#     bestAccuraccy = 0

#     print(f"Epoch {t+1}\n-------------------------------")

#     if (t + 1) % 10 == 0:
#         checkpoint = {'stateDict' : classifier.state_dict(), 'optimiser' : optimiser.state_dict}
#         Savecheckpoint(checkpoint)


#     training(model=classifier, trainDataLoader=RealtrainDataLoader, lossFunction=lossFunction, optimiser=optimiser)

#     if (t + 1) % 2 == 0:
#         accuraccy = testing(classifier, validationDataLoader, lossFunction)

#         if (accuraccy > bestAccuraccy):
#             bestAccuraccy = accuraccy
#             Savecheckpoint(classifier, (t+1), optimiser, bestAccuraccy)

print("Done")



##Removed dropout in convolutional layers, increased dropout to 0.5, reduced strides to 1 for all max pooling.
##Added to more conv layers, and reduced output sizes for existing ones
##

##Testing every 5th epoch

# Plan:

# 1. Transformation into same size


# 2. Filters

# 3. Max Pooling

# 4. flatten

# 5. Final Layer

# 6. Go back to add data augmentation

# 7. Implement regularization

# 8. implement ways to prevent overfitting






