import numpy as np
import torch

from sobolt.gorgan.nn import OrientationMagnitudeExtractor


def canny_edge_extractor(img):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # a sobel edges
    sobel_detector = OrientationMagnitudeExtractor(device)

    # Get frequency distribution for magnitude & orientations
    grad_mag, grad_orient = sobel_detector(img)
    grad_mag, grad_orient = grad_mag.cpu().numpy(), grad_orient.cpu().numpy()
    # b non-maximum suppression
    grad_mag = np.array(grad_mag).squeeze()
    grad_orient = np.array(grad_orient).squeeze()
    #    plt.imsave('sobel_edges.png', grad_mag)
    grad_mag = non_max_suppression(grad_mag, grad_orient)

    #    plt.imsave('grad_mag_supressed.png', grad_mag)

    # c apply thresholds for irrelevant, weak, and strong
    grad_mag, weak, strong = threshold(grad_mag, low_threshold=0.01, high_threshold=0.03)
    #    plt.imsave('grad_mag_thresholded.png', grad_mag*255)

    # d hysteresis for edge tracking to divide weak into and irrelevant
    canny_edges = hysteresis(grad_mag, weak)
    #    plt.imsave('canny_edges.png', canny_edges)

    return canny_edges


def non_max_suppression(grad_mag, D):
    """calculates the maxima in img for each gradient/edge using
    the gradient orientations in D, and sets other values to zero.
    """
    M, N = grad_mag.shape
    grad_mag_suppressed = np.zeros((M, N), dtype=np.float32)
    angle = D * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255
                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = grad_mag[i, j + 1]
                    r = grad_mag[i, j - 1]
                # angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = grad_mag[i + 1, j - 1]
                    r = grad_mag[i - 1, j + 1]
                # angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = grad_mag[i + 1, j]
                    r = grad_mag[i - 1, j]
                # angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = grad_mag[i - 1, j - 1]
                    r = grad_mag[i + 1, j + 1]

                if (grad_mag[i, j] >= q) and (grad_mag[i, j] >= r):
                    grad_mag_suppressed[i, j] = grad_mag[i, j]
                else:
                    grad_mag_suppressed[i, j] = 0

            except IndexError as e:
                pass
    return grad_mag_suppressed


def threshold(img, low_threshold=0.01, high_threshold=0.03):

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.float32)

    weak = 0.5
    strong = 1.0

    strong_i, strong_j = np.where(img >= high_threshold)
    zeros_i, zeros_j = np.where(img < low_threshold)

    weak_i, weak_j = np.where((img <= high_threshold) & (img >= low_threshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res, weak, strong)


def hysteresis(img, weak, strong=255):
    M, N = img.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if img[i, j] == weak:
                try:
                    if (
                        (img[i + 1, j - 1] == strong)
                        or (img[i + 1, j] == strong)
                        or (img[i + 1, j + 1] == strong)
                        or (img[i, j - 1] == strong)
                        or (img[i, j + 1] == strong)
                        or (img[i - 1, j - 1] == strong)
                        or (img[i - 1, j] == strong)
                        or (img[i - 1, j + 1] == strong)
                    ):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


# if __name__ == "__main__":
#
#     img = cv2.imread("/home/otto/Downloads/batch20, valimg 1 of 3.png")
#     img = img.astype(np.float32) / 255
#     split = np.split(img, 3, axis=1)
#
#     img_o_np = split[0][:128, :128, :]
#     img_g_np = split[0]
#     img_o_np = img_o_np.swapaxes(1, 2).swapaxes(1, 0)
#     img_g_np = img_g_np.swapaxes(1, 2).swapaxes(1, 0)
#     img_o = np.expand_dims(img_o_np, 0)
#     img_g = np.expand_dims(img_g_np, 0)
#     img_o = torch.from_numpy(img_o)
#     img_g = torch.from_numpy(img_g)
#     canny_score = calc_canny_density_ratio(img_g, img_o)
#
#     print(canny_score)
