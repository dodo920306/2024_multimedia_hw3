from cv2 import imread, cvtColor, COLOR_BGR2RGB, COLOR_BGR2GRAY, COLOR_GRAY2RGB, GaussianBlur, imshow, normalize, NORM_MINMAX, threshold, THRESH_BINARY, namedWindow, createTrackbar, imwrite
from sys import argv
from matplotlib.pyplot import subplots, tight_layout, savefig, show
from numpy import histogram, array, zeros, zeros_like, where, sum, round
from numpy.ma import masked_equal, filled

def apply_threshold(lb):
    global img
    _, timg = threshold(img, lb, 255, THRESH_BINARY)
    imshow('Edge Detection', timg)
    # imwrite('result2.png', timg)

if __name__ == '__main__':
    cimg = imread(argv[1])
    assert cimg is not None, "file could not be read, check with os.path.exists()"

    _, axes = subplots(2, 2, figsize=(10, 5))
    axes[0][0].imshow(cvtColor(cimg, COLOR_BGR2RGB))
    axes[0][0].set_title('Original Colored Image')
    axes[0][0].axis('off')

    cimg = cvtColor(cimg, COLOR_BGR2GRAY)

    # bins = 0 ~ 256
    # so x = 0 ~ 256
    y, x = histogram(cimg.flatten(), bins=range(257))
    # x will have one element more than y, and that element is the upper bound. In this case, x[-1] = 256.
    # This is because the last bin is [255, 256) to include 255.
    axes[1][0].bar(x[:-1], y)
    axes[1][0].set_xlabel('Gray Scale')
    axes[1][0].set_ylabel('Frequency')
    axes[1][0].set_title('Histogram of Original Grayscale')

    # Histogram Equalization
    pdf = y / sum(y)
    y_normalized = pdf * 255
    cdf = round(y_normalized.cumsum()).astype('uint8')

    # Now, cdf can act like a map, so the Histogram Equalization of cimg is cdf[cimg].
    img = cdf[cimg]
    axes[0][1].imshow(cvtColor(img, COLOR_GRAY2RGB))
    axes[0][1].set_title('Equalized')
    axes[0][1].axis('off')

    y, x = histogram(img.flatten(), bins=range(257))
    axes[1][1].bar(x[:-1], y)
    axes[1][1].set_xlabel('Gray Scale')
    axes[1][1].set_ylabel('Frequency')
    axes[1][1].set_title('Histogram of Equalized')

    # Edge Detection
    sobel_x = array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = sobel_x.T

    img = GaussianBlur(cimg, (3, 3), 0)
    zero_padding = zeros((img.shape[0] + 2, img.shape[1] + 2)).astype('int32')
    zero_padding[1:-1, 1:-1] = img
    res = zeros_like(img).astype('int32')

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            res_x = sum(zero_padding[x : x + 3, y : y + 3] * sobel_x) ** 2
            res_y = sum(zero_padding[x : x + 3, y : y + 3] * sobel_y) ** 2
            res[x, y] = (res_x + res_y) ** 0.5

    img = normalize(where(res > 255, 255, res).astype('uint8'), None, 0, 255, NORM_MINMAX)
    namedWindow('Edge Detection')
    createTrackbar('Lower Bound:', 'Edge Detection', 0, 255, apply_threshold)
    apply_threshold(0)

    tight_layout()
    savefig("result1.png")
    show()
