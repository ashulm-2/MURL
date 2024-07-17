from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

def imageprepare(file):
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

  tv = list(newImage.getdata())  # get pixel values

  # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
  tva = [(255 - x) * 1.0 / 255.0 for x in tv]
  #print(tva)
  return tva

#x=imageprepare('testing-zoom.png')#file path here
#print(len(x))# mnist IMAGES are 28x28=784 pixels

