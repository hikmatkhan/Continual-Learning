import torch, torchvision

resnet = torchvision.models.resnet18(pretrained=True)
labels = torch.zeros(1, 1000)
labels[0][0] = 1
# print(labels)
sample = torch.randn((1, 3, 64, 64))
cel = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(resnet.parameters(), lr=0.001)
for i in range(0, 10):

    predictions = resnet(sample)
    # print("Label shape:", labels.shape)
    # print("Prediction shape:", predictions.shape)
    # print("Label:", labels)
    # print("Prediction:", predictions)
    loss = cel(labels, labels)
    print("Loss:", loss)
    loss.backward()
    optim.zero_grad()
    optim.step()

