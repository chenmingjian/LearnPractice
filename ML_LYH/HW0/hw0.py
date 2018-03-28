from PIL import Image

im = Image.open('data/westbrook.jpg')
w, h = im.size
im = im.convert('RGB');
src = im.load()
for i in range(w):
	for j in range(h):
		src[i,j] = (src[i,j][0]//2, src[i,j][1]//2, src[i,j][2]//2)

im.save('ouput.jpg','jpeg')