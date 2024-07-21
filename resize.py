from PIL import Image, ImageFilter
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.PILToTensor()])

def CropImage(file):
  im = Image.open(file).convert('L') #opens in grayscale
  width,height = im.size
  print(width,height)
  #im.show()
  PixIm = im.load()

  #first we trim the left
  Found = False
  for x in range(width):
    for y in range(height):
      if PixIm[x,y] < 200:
        Found = True
        break
    if Found:
      TrimLeft = x
      break
  
  #now we trim the right
  Found = False
  for x in range(width-1,0,-1):
    for y in range(height-1,0,-1):
      if PixIm[x,y] < 200:
        Found = True
        break
    if Found:
      TrimRight = x
      break


  #first we trim the top
  Found = False
  for y in range(height):
    for x in range(width): 
      if PixIm[x,y] < 200:
        Found = True
        break
    if Found:
      TrimTop = y
      break
  
  #now we trim the bottom
  Found = False
  for y in range(height-1,0,-1):
    for x in range(width-1,0,-1):
      if PixIm[x,y] < 200:
        Found = True
        break
    if Found:
      TrimBottom = y
      break
  

  im1 = im.crop((TrimLeft,TrimTop,TrimRight,TrimBottom))
  im1.save("trimmed-" + file)
  #im1.show()
        

CropImage("testing3.png")




def ImagePrep(file):
  """
  This function returns the pixel values.
  The imput is a png file location.
  """
  im = Image.open(file).convert('L') #opens in grayscale
  width = float(im.size[0])
  height = float(im.size[1])
  newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

  if width > height:  # check which dimension is bigger
    # Width is bigger. Width becomes 20 pixels.
    nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
    if (nheight == 0):  # rare case but minimum is 1 pixel
      nheight = 1
    # resize and sharpen
    img = im.resize((20, nheight), Image.LANCZOS).filter(ImageFilter.SHARPEN)
    wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
    newImage.paste(img, (4, wtop))  # paste resized image on white canvas
  else:
    # Height is bigger. Heigth becomes 20 pixels.
    nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
    if (nwidth == 0):  # rare case but minimum is 1 pixel
      nwidth = 1
    # resize and sharpen
    img = im.resize((nwidth, 20), Image.LANCZOS).filter(ImageFilter.SHARPEN)
    wleft = int(round(((28 - nwidth) / 2), 0))  # calculate vertical pozition
    newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

  #to view the image
  #plt.imshow(newImage,cmap="binary")
  #plt.show()

  TT = transform(newImage)
  TT = (255-TT)*1.0 / 255.0
  return torch.unsqueeze(TT,0)

  
  """
  tv = list(newImage.getdata())  # get pixel values
  # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
  tva = [(255 - x) * 1.0 / 255.0 for x in tv]
  #print(tva)
  return tva
  """

#x = ImagePrep('testing-zoom.png')#file path here
#print(x.shape)# mnist IMAGES are 28x28=784 pixels

