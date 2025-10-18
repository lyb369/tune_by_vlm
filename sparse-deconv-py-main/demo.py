
from sparse_recon.sparse_deconv import sparse_deconv
from skimage import io
from matplotlib import pyplot as plt

if __name__ == '__main__':
    im = io.imread('test.tif')
    plt.imshow(im, cmap = 'gray')
    plt.show()
    pixelsize = 65 #(nm)
    resolution = 280 #(nm)
    # img_recon = sparse_deconv(im, resolution / pixelsize)
    img_recon = sparse_deconv(
        im,
        resolution / pixelsize,     # 或 [sx, sy] / [sx, sy, sz]
        sparse_iter=100,
        fidelity=15,
        sparsity=1,
        tcontinuity=0.5,
        background=2,
        deconv_iter=7,
        deconv_type=1,
        up_sample=0
    )
    plt.imshow(img_recon / img_recon.max() * 255, cmap = 'gray')
    plt.show()
    io.imsave('test_processed.tif', img_recon.astype(im.dtype))