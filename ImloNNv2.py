import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.transforms.v2 import ToTensor
import scipy

device = "cpu"

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
    device = "cuda"

else:
    print("No GPU available. Training will run on CPU.")



trainingData = datasets.Flowers102(
    root = "ImageData",
    split = "train",
    download = True,
    transform = v2.Compose([
    
        v2.Resize((224,224), antialias=True),
        v2.RandomHorizontalFlip(),
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

Realvalidation = datasets.Flowers102(
    root = "ImageData",
    split = "val",
    download = True,
        transform = v2.Compose([
        v2.ToTensor(),
        v2.Resize((224,224), antialias=True),
        v2.Normalize(torch.Tensor(meanCalc), torch.Tensor(stdCalc))
        
    ]),
)

RealtrainDataLoader = DataLoader(Realtraining, batch_size=batchSize, shuffle=True)
RealvalidationDataLoader = DataLoader(Realvalidation, batch_size=batchSize, shuffle=True)

class CNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.feature = nn.Sequential(

            nn.Conv2d(3, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3, stride=4),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # nn.MaxPool2d(kernel_size=3, stride=4),
            # nn.ReLU(),

            

            nn.Flatten(),

        )

        Nchannels = self.feature(torch.empty(1, 3, 224, 224)).size(-1)

        self.classify = nn.Sequential(

            nn.Linear(int(Nchannels), int(Nchannels/128)),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(int(Nchannels/128), int(102)),
              

        )

        
        

    def forward(self, x):
        features = self.feature(x)
        
        return self.classify(features)

        
    
classifier = CNN().to(device)

optimiser = Adam(classifier.parameters(), lr=1e-4, weight_decay=1e-5)

lossFunction = nn.CrossEntropyLoss()


def training(model, trainDataLoader, lossFunction, optimiser):

    model.train()

    for batch, (X,y) in enumerate(trainDataLoader):

        if device == "cuda":
            X,y = X.cuda(), y.cuda()

        prediction = model(X)
        loss = lossFunction(prediction, y)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
       

        loss, current = loss.item(), batch * batchSize + len(X)
        
        print(f"loss: {loss:>7f}  [{current:>5d}/{len(trainDataLoader):>5d}]")

bestAcc = 1

def testing(model, testDataLoader, lossFunction):
    model.eval()

    size = len(testDataLoader.dataset)
    numberOfBatches = len(testDataLoader)

    testLoss = 0
    correct = 0

    with torch.no_grad():

        for X, y in testDataLoader:

            if device == "cuda":
                X,y = X.cuda(), y.cuda()


            prediction = model(X)
            testLoss += lossFunction(prediction, y).item()
            correct += (prediction.argmax(1) == y).type(torch.float).sum().item()

        testLoss /= numberOfBatches
        correct /= size

        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {testLoss:>8f} \n")

        lossFunction = nn.CrossEntropyLoss()

        checkpoint = torch.load('30epochRun.pth.tar')

        accu = (checkpoint['BestAcc'])

        if ((100 * correct) > accu) & ((100 * correct) > 35):
            bestAcc = (100 * correct)
            print("Saving....")

            state = {
                'epoch' : t,
                'model' : classifier.state_dict(),
                'BestAcc' : bestAcc,
                'optimiser' : optimiser.state_dict(),
            }

            torch.save(state, '30epochRun2.pth.tar')

checkpoint = torch.load('30epochRun.pth.tar')

print(checkpoint['BestAcc'])

epochs = 80


for t in range(epochs):

    print(f"Epoch {t+1}\n-------------------------------")

    training(model=classifier, trainDataLoader=RealtrainDataLoader, lossFunction=lossFunction, optimiser=optimiser)

    
    testing(classifier, RealvalidationDataLoader, lossFunction)

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

# 7. Implement normalisation

# 8. implement ways to prevent overfitting






# import torch
# from torch import nn
# from torch.utils.data import DataLoader
# from torch.optim import Adam
# from torchvision import datasets
# from torchvision.transforms import v2
# from torchvision.transforms.v2 import ToTensor
# import scipy






# trainingData = datasets.Flowers102(
#     root = "ImageData",
#     split = "train",
#     download = True,
#     transform = v2.Compose([
    
#         v2.Resize((224,224), antialias=True),
#         v2.RandomHorizontalFlip(),
#         v2.ToTensor(),
#         #v2.Normalize(torch.Tensor(mean), torch.Tensor(std))

#     ]),
# )


# testing = datasets.Flowers102(
#     root = "ImageData",
#     split = "test",
#     download = True,
#         transform = v2.Compose([

#         v2.Resize((224,224), antialias=True),
#         v2.ToTensor(),
#         #v2.Normalize(torch.Tensor(mean), torch.Tensor(std)),
        
#     ]),
# )

# validation = datasets.Flowers102(
#     root = "ImageData",
#     split = "val",
#     download = True,
#         transform = v2.Compose([
#         v2.ToTensor(),
#         v2.Resize((224,224), antialias=True),
        
#     ]),
# )

# # print(testing)
# # print(testing)
# # print(validation)

# batchSize = 64
# dropOut = 0.5

# trainDataLoader = DataLoader(trainingData, batch_size=batchSize, shuffle=True)
# testDataLoader = DataLoader(testing, batch_size=batchSize)
# validationDataLoader = DataLoader(validation, batch_size=batchSize)
# def getMeanAndSTD():
#     mean = 0
#     std = 0

#     totalImages = 0

#     for images, _ in trainDataLoader:
#         imageBatchCount = images.size(0)
#         images = images.view(imageBatchCount, images.size(1), -1)
#         mean += images.mean(2).sum(0)
#         std += images.std(2).sum(0)
#         totalImages += imageBatchCount

#     mean /= totalImages
#     std /= totalImages

#     return mean, std

# meanCalc, stdCalc = getMeanAndSTD()


# Realtraining = datasets.Flowers102(
#     root = "ImageData",
#     split = "train",
#     download = True,
#     transform = v2.Compose([
    
#         v2.RandomResizedCrop((224,224), antialias=True),
#         v2.RandomHorizontalFlip(),
#         v2.RandomRotation(10),
#         v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
#         v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
#         v2.RandomAdjustSharpness(sharpness_factor=2),
#         v2.ToTensor(),
#         #v2.RandomPerspective(0.3, 0.5),
#         v2.Normalize(torch.Tensor(meanCalc), torch.Tensor(stdCalc))

#     ]),
# )

# Realvalidation = datasets.Flowers102(
#     root = "ImageData",
#     split = "val",
#     download = True,
#         transform = v2.Compose([
#         v2.ToTensor(),
#         v2.Resize((224,224), antialias=True),
#         v2.Normalize(torch.Tensor(meanCalc), torch.Tensor(stdCalc))
        
#     ]),
# )

# RealtrainDataLoader = DataLoader(Realtraining, batch_size=batchSize, shuffle=True)
# RealvalidationDataLoader = DataLoader(Realvalidation, batch_size=batchSize, shuffle=True)

# class CNN(nn.Module):
    
#     def __init__(self):
#         super().__init__()
        
#         self.feature = nn.Sequential(

#             nn.Conv2d(3, 16, kernel_size=3, stride=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),

#             nn.Conv2d(16, 32, 3),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),

#             nn.Conv2d(32, 32, 3),
#             nn.ReLU(),

#             nn.MaxPool2d(kernel_size=3, stride=4),
#             nn.ReLU(),

#             nn.Conv2d(32, 32, 3),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),

#             nn.Conv2d(32, 64, 3),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),

#             # nn.MaxPool2d(kernel_size=3, stride=4),
#             # nn.ReLU(),

            

#             nn.Flatten(),

#         )

#         Nchannels = self.feature(torch.empty(1, 3, 224, 224)).size(-1)

#         self.classify = nn.Sequential(

#             nn.Linear(int(Nchannels), int(Nchannels/128)),
#             nn.ReLU(),
#             nn.Dropout(0.25),
#             nn.Linear(int(Nchannels/128), int(102)),
              

#         )

        
        

#     def forward(self, x):
#         features = self.feature(x)
        
#         return self.classify(features)

        
    
# classifier = CNN().to("cuda")

# optimiser = Adam(classifier.parameters(), lr=1e-4)

# lossFunction = nn.CrossEntropyLoss()


# def training(model, trainDataLoader, lossFunction, optimiser):

#     model.train()

#     for batch, (X,y) in enumerate(trainDataLoader):

#         X,y = X.cuda(), y.cuda()

#         prediction = model(X)
#         loss = lossFunction(prediction, y)

#         optimiser.zero_grad()
#         loss.backward()
#         optimiser.step()
       

#         loss, current = loss.item(), batch * batchSize + len(X)
        
#         print(f"loss: {loss:>7f}  [{current:>5d}/{len(trainDataLoader):>5d}]")

# def testing(model, testDataLoader, lossFunction):
#     model.eval()

#     size = len(testDataLoader.dataset)
#     numberOfBatches = len(testDataLoader)

#     testLoss = 0
#     correct = 0

#     with torch.no_grad():

#         for X, y in testDataLoader:
#             X, y = X.cuda(), y.cuda()
#             prediction = model(X)
#             testLoss += lossFunction(prediction, y).item()
#             correct += (prediction.argmax(1) == y).type(torch.float).sum().item()

#         testLoss /= numberOfBatches
#         correct /= size

#         print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {testLoss:>8f} \n")

#         lossFunction = nn.CrossEntropyLoss()



# epochs = 20

# for t in range(epochs):

#     print(f"Epoch {t+1}\n-------------------------------")

#     training(model=classifier, trainDataLoader=RealtrainDataLoader, lossFunction=lossFunction, optimiser=optimiser)

    
#     testing(classifier, RealvalidationDataLoader, lossFunction)

# print("Done")



# ##Removed dropout in convolutional layers, increased dropout to 0.5, reduced strides to 1 for all max pooling.
# ##Added to more conv layers, and reduced output sizes for existing ones
# ##

# ##Testing every 5th epoch

# # Plan:

# # 1. Transformation into same size


# # 2. Filters

# # 3. Max Pooling

# # 4. flatten

# # 5. Final Layer

# # 6. Go back to add data augmentation

# # 7. Implement normalisation

# # 8. implement ways to prevent overfitting






