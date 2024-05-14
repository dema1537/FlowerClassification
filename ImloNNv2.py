import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.transforms.v2 import ToTensor
import scipy
import matplotlib.pyplot as plt



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

        v2.Resize((224,224), antialias=True),
        v2.ToTensor(),

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

        v2.Resize((224,224), antialias=True),
        v2.RandomHorizontalFlip(),
       
        #v2.RandomVerticalFlip(),
        v2.RandomRotation(45),
        # v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        # v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
        v2.RandomAdjustSharpness(sharpness_factor=2),
        v2.ElasticTransform(alpha=25.0),
        v2.ToTensor(),
        v2.RandomPerspective(0.1, 0.5),
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
RealvalidationDataLoader = DataLoader(Realvalidation, batch_size=batchSize, shuffle=True)

for x,y in RealtrainDataLoader:
  x = x.to(device)
  fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(12,8))
  for i in range(2):
    for j in range(4):
      ax[i,j].imshow(x[i*4+j].cpu().permute(1,2,0))
      ax[i,j].axis('off')
  break

plt.show()

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

            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=3),
            nn.ReLU(),

            nn.Conv2d(64, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # nn.Conv2d(128, 256, 3),
            # nn.ReLU(),




            # nn.Conv2d(256, 512, 3),
            # nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),

            nn.Flatten(),
            #nn.Dropout(0.5)


        )

        Nchannels = self.feature(torch.empty(1, 3, 224, 224)).size(-1)
        print(Nchannels)

        self.classify = nn.Sequential(


            nn.Linear(int(Nchannels), int(Nchannels * 2)),
            nn.ReLU(),
            #nn.Dropout(0.5),

            # nn.Linear(int(Nchannels * 2), int(Nchannels)),
            # nn.ReLU(),
            # nn.Dropout(0.5),


            nn.Linear(int(Nchannels * 2), int(102)),

        )




    def forward(self, x):
        features = self.feature(x)

        return self.classify(features)



classifier = CNN().to(device)

lossFunction = nn.CrossEntropyLoss()


optimiser = Adam(classifier.parameters(), lr=1e-4, weight_decay=1e-2)

#optimiser = torch.optim.SGD(classifier.parameters(), lr=1e-2, weight_decay=1e-3)




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



        print(f"loss: {loss:>7f}  [{current:>5d}/{len(trainDataLoader.dataset):>5d}]")



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



# checkpoint = torch.load('30epochRun.pth.tar')

#         accu = (checkpoint['BestAcc'])

        accu = 30

        # if ((100 * correct) > accu) & ((100 * correct) > 35):
        #     accu = (100 * correct)
        #     print("Saving....")

        #     state = {
        #         'epoch' : t,
        #         'model' : classifier.state_dict(),
        #         'BestAcc' : bestAcc,
        #         'optimiser' : optimiser.state_dict(),
        #     }

        #     torch.save(state, '30epochRun2.pth.tar')

# checkpoint = torch.load('30epochRun.pth.tar')

# print(checkpoint['BestAcc'])

epochs = 60


for t in range(epochs):

    print(f"Epoch {t+1}\n-------------------------------")

    training(model=classifier, trainDataLoader=RealtrainDataLoader, lossFunction=lossFunction, optimiser=optimiser)


    testing(classifier, RealvalidationDataLoader, lossFunction)

print("Done")
