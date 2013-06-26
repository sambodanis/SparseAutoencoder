import Image, numpy

im = Image.open("test_image_1.png")

size_tup = im.size

# print size_tup
for x in size_tup:
	print x
gray_pixels = []
for (r, g, b) in im.getdata():
	gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
	gray_pixels.append(gray)
	print gray
	# print str(r) + "," + str(g) + "," + str(b)



# grayscale_image = im.convert('1')
# grayscale_image.show()
# for pixel in im.getdata():
# 	print pixel