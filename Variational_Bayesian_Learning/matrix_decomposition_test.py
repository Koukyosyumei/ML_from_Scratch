import numpy as np
from PIL import Image

from matrix_decomposition import EVBMD, SVD, VBMD

if __name__ == "__main__":

    img = Image.open("Variational_Bayesian_Learning/data/stop.jpeg")
    gray_img = img.convert('L')
    v = np.asarray(gray_img) / 255

    h = 10
    svd = SVD()
    svd.fit(v, h=h)
    u = svd.reconstruct()
    u_img = Image.fromarray(np.uint8(u * 255))
    u_img.save("stop_reconstructed_with_svd.jpg")

    sigma = 0.1
    ca_square = np.array([1e8]*h)
    cb_square = np.array([1e8]*h)
    vbmd = VBMD(sigma=sigma, ca_square=ca_square, cb_square=cb_square)
    vbmd.fit(v, h=h)
    u = vbmd.reconstruct()
    u_img = Image.fromarray(np.uint8(u * 255))
    u_img.save("stop_reconstructed_with_vbmd.jpg")

    evbmd = EVBMD()
    evbmd.fit(v)
    u = evbmd.reconstruct()
    u_img = Image.fromarray(np.uint8(u * 255))
    u_img.save("stop_reconstructed_with_evbmd.jpg")
