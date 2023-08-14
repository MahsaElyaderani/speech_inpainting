import numpy as np

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    #l1 = np.mean(np.abs((original-compressed)*mask))
    if (mse == 0):  # MSE is zero means no noise is present in the signal, therefore PSNR have no importance.
        return 100
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    return psnr, mse

