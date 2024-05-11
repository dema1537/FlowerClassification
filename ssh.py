import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.transforms.v2 import ToTensor
import scipy

from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle





if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available. Training will run on CPU.")



trainingData = datasets.Flowers102(
    root = "ImageData",
    split = "train",
    download = True,
    transform = v2.Compose([

        v2.Resize((224,224), antialias=True),
        v2.ToTensor(),

    ]),
)


testing = datasets.Flowers102(
    root = "ImageData",
    split = "test",
    download = True,
        transform = v2.Compose([

        v2.Resize((224,224), antialias=True),
        v2.ToTensor(),

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


batchSize = 64
learningRate = 0.001
weightDecay = 0.001
dropOut = 0.5

trainDataLoader = DataLoader(trainingData, batch_size=batchSize, shuffle=True, pin_memory=True)
testDataLoader = DataLoader(testing, batch_size=batchSize, pin_memory=True)
validationDataLoader = DataLoader(validation, batch_size=batchSize, pin_memory=True)

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
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
        v2.RandomAdjustSharpness(sharpness_factor=2),
        v2.ToTensor(),
        #v2.RandomPerspective(0.3, 0.5),
        v2.Normalize(torch.Tensor(meanCalc), torch.Tensor(stdCalc))

    ]),
)


def getMeanAndSTD():
    mean = 0
    std = 0

    totalImages = 0

    for images, _ in validationDataLoader:
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



Realvalidation = datasets.Flowers102(
    root = "ImageData",
    split = "val",
    download = True,
    transform = v2.Compose([

        v2.Resize((224,224), antialias=True),
        #v2.RandomPerspective(0.3, 0.5),
        v2.ToTensor(),
        v2.Normalize(torch.Tensor(meanCalc), torch.Tensor(stdCalc))

    ]),
)



RealtrainDataLoader = DataLoader(Realtraining, batch_size=batchSize, shuffle=True, pin_memory=True)
RealvalidationDataLoader = DataLoader(Realvalidation, batch_size=batchSize, shuffle=True, pin_memory=True)


class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.feature = nn.Sequential(




            nn.Conv2d(3, 16, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3),
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
            #nn.BatchNorm1d(int(Nchannels/128)),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(int(Nchannels/128), int(Nchannels/128)),
            #nn.BatchNorm1d(int(Nchannels/128)),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(int(Nchannels/128), int(102)),




        )




    def forward(self, x):
        features = self.feature(x)

        return self.classify(features)



classifier = CNN().to("cuda")


tensor = torch.randn((3, 3))
tensor = tensor.to('cuda')

optimiser = Adam(classifier.parameters(), lr=learningRate, weight_decay=weightDecay, betas=(0.9, 0.999))

lossFunction = nn.CrossEntropyLoss()

l1 = 0.001


def training(model, trainDataLoader, lossFunction, optimiser):

    model.train()

    for batch, data in enumerate(trainDataLoader, 0):

        optimiser.zero_grad()

        X, y = data[0].cuda(), data[1].cuda()
        prediction = model(X)
        loss = lossFunction(prediction, y)

        # l1Regularisation = 0.0
        # for param in model.parameters():
        #     l1Regularisation += torch.norm(param, p=1)

        # loss += l1 * l1Regularisation.detach().item()


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
            X, y = X.cuda(), y.cuda()
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
        'optimiser' : optimiser.state_dict(),
    }

    torch.save(state, 'SimpleModeltests.pth.tar')

def Loadcheckpoint(checkpoint):
    print("Loading...")

    classifier.load_state_dict(checkpoint['model'])
    #optimiser.load(checkpoint['optimiser'])


# loadModel = False

# if loadModel == True:
#     Loadcheckpoint(torch.load("SimpleModel370.pth.tar"))

#     print("Testing...")

#     testing(classifier, testDataLoader, lossFunction)

epochs = 80

bestAccuraccy = 0

for t in range(epochs):



    print(f"Epoch {t+1}\n-------------------------------")



    training(model=classifier, trainDataLoader=RealtrainDataLoader, lossFunction=lossFunction, optimiser=optimiser)

    if (t + 1) % 2 == 0:
        accuraccy = testing(classifier, RealvalidationDataLoader, lossFunction)
        print("Best: " + str(bestAccuraccy))

        if (accuraccy > bestAccuraccy):
            bestAccuraccy = accuraccy
        #     Savecheckpoint(classifier, (t+1), optimiser, bestAccuraccy)

        #Savecheckpoint(classifier, (t+1), optimiser, bestAccuraccy)

Savecheckpoint(classifier, (t+1), optimiser, bestAccuraccy)

print("Done")

#Approx 13mins for 10 epochs
#1.3 mins per epoch
#8 * 60 = 480
#~370 epochs in 8 hrs

#last 10 epochs: 16mins 15

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






