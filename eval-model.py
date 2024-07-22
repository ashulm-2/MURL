from cnn_mnist_yt import CNN
from cnn_emnist_yt import CNNE
from resize import ImagePrep
import torch

#model = CNN()
#model.load_state_dict(torch.load("CNN-weights.pt"))

model = CNNE()
model.load_state_dict(torch.load("CNNE-weights.pt"))
model.eval()

#Im = torch.tensor(torch.zeros([1,1,28,28]))
Im = ImagePrep('test.png')
yHat = model(Im)
print(yHat)
print(torch.argmax(yHat).item())
