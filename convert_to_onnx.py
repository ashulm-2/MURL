from cnn_mnist_yt import CNN
import torch

def main():
  model = CNN()
  model.load_state_dict(torch.load("CNN-weights.pt"))
  model.eval()
  dummy_input = torch.zeros(1,1,28,28)
  torch.onnx.export(model,dummy_input,"onnx_model.onnx", verbose=True)
  
if __name__ == "__main__":
  main()