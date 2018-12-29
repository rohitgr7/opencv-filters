import numpy as np
import cv2
from scipy.interpolate import UnivariateSpline


def _create_LUT_BUC1(x, y):
    spl = UnivariateSpline(x, y)
    return spl(range(256))


def _create_loopup_tables():
    incr_ch_lut = _create_LUT_BUC1(
        [0, 64, 128, 192, 256], [0, 70, 140, 210, 256])
    decr_ch_lut = _create_LUT_BUC1(
        [0, 64, 128, 192, 256], [0, 30, 80, 120, 192])

    return incr_ch_lut, decr_ch_lut


def _warming(orig):
    incr_ch_lut, decr_ch_lut = _create_loopup_tables()

    c_b, c_g, c_r = cv2.split(orig)
    c_r = cv2.LUT(c_r, incr_ch_lut).astype(np.uint8)
    c_b = cv2.LUT(c_b, decr_ch_lut).astype(np.uint8)
    img = cv2.merge((c_b, c_g, c_r))

    H, S, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    S = cv2.LUT(S, incr_ch_lut).astype(np.uint8)

    output = cv2.cvtColor(cv2.merge((H, S, V)), cv2.COLOR_HSV2BGR)
    return output


def _cooling(orig):
    incr_ch_lut, decr_ch_lut = _create_loopup_tables()

    c_b, c_g, c_r = cv2.split(orig)
    c_r = cv2.LUT(c_r, decr_ch_lut).astype(np.uint8)
    c_b = cv2.LUT(c_b, incr_ch_lut).astype(np.uint8)
    img = cv2.merge((c_b, c_g, c_r))

    H, S, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    S = cv2.LUT(S, decr_ch_lut).astype(np.uint8)

    output = cv2.cvtColor(cv2.merge((H, S, V)), cv2.COLOR_HSV2BGR)
    return output


def _cartoon2(orig):
    img = orig.copy()

    for _ in range(2):
        img = cv2.pyrDown(img)

    for _ in range(7):
        img = cv2.bilateralFilter(img, 9, 9, 7)

    for _ in range(2):
        img = cv2.pyrUp(img)

    img_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)
    img_edge = cv2.adaptiveThreshold(
        img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR)

    output = cv2.bitwise_and(img, img_edge)
    return output


def _cartoon(orig):
    img = np.copy(orig)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=5)
    edges = 255 - edges
    ret, edge_mask = cv2.threshold(edges, 150, 255, cv2.THRESH_BINARY)
    img_bilateral = cv2.edgePreservingFilter(
        img, flags=2, sigma_s=50, sigma_r=0.4)
    output = np.zeros(img_gray.shape)
    output = cv2.bitwise_and(img_bilateral, img_bilateral, mask=edge_mask)
    return output


def _color_dodge(top, bottom):
    output = cv2.divide(bottom, 255 - top, scale=256)
    return output


def _sketch_pencil_using_blending(orig, kernel_size=21):
    img = np.copy(orig)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_inv = 255 - img_gray
    img_gray_inv_blur = cv2.GaussianBlur(
        img_gray_inv, (kernel_size, kernel_size), 0)
    output = _color_dodge(img_gray_inv_blur, img_gray)
    return cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)


def _sketch_pencil_using_edge_detection(orig):
    img = np.copy(orig)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    edges = cv2.Laplacian(img_gray_blur, cv2.CV_8U, ksize=5)
    edges = 255 - edges
    ret, edge_mask = cv2.threshold(edges, 150, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR)


def _adjust_contrast(orig, scale_factor):
    img = np.copy(orig)
    ycb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycb_img = np.float32(ycb_img)
    y_channel, Cr, Cb = cv2.split(ycb_img)
    y_channel = np.clip(y_channel * scale_factor, 0, 255)
    ycb_img = np.uint8(cv2.merge([y_channel, Cr, Cb]))
    img = cv2.cvtColor(ycb_img, cv2.COLOR_YCrCb2BGR)
    return img


def _apply_vignette(orig, vignette_scale):
    img = np.copy(orig)
    img = np.float32(img)
    rows, cols = img.shape[:2]
    k = np.min(img.shape[:2]) / vignette_scale
    kernel_x = cv2.getGaussianKernel(cols, k)
    kernel_y = cv2.getGaussianKernel(rows, k)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)

    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    img[:, :, 0] += img[:, :, 0] * mask
    img[:, :, 1] += img[:, :, 1] * mask
    img[:, :, 2] += img[:, :, 2] * mask
    img = np.clip(img / 2, 0, 255)
    return np.uint8(img)


def _xpro2(orig, vignette_scale=3):
    img = np.copy(orig)
    img = _apply_vignette(img, vignette_scale)
    b_channel = img[:, :, 0]
    g_channel = img[:, :, 1]
    r_channel = img[:, :, 2]
    orig_r = np.array([0, 42, 105, 148, 185, 255])
    orig_g = np.array([0, 40, 85, 125, 165, 212, 255])
    orig_b = np.array([0, 40, 82, 125, 170, 225, 255])
    r_curve = np.array([0, 28, 100, 165, 215, 255])
    g_curve = np.array([0, 25, 75, 135, 185, 230, 255])
    b_curve = np.array([0, 38, 90, 125, 160, 210, 222])
    full_range = np.arange(0, 256)
    b_LUT = np.interp(full_range, orig_b, b_curve)
    g_LUT = np.interp(full_range, orig_g, g_curve)
    r_LUT = np.interp(full_range, orig_r, r_curve)
    b_channel = cv2.LUT(b_channel, b_LUT)
    g_channel = cv2.LUT(g_channel, g_LUT)
    r_channel = cv2.LUT(r_channel, r_LUT)
    img[:, :, 0] = np.uint8(b_channel)
    img[:, :, 1] = np.uint8(g_channel)
    img[:, :, 2] = np.uint8(r_channel)
    img = _adjust_contrast(img, 1.2)
    return img


def _clarendon(orig):
    img = np.copy(orig)
    b_channel = img[:, :, 0]
    g_channel = img[:, :, 1]
    r_channel = img[:, :, 2]
    x_values = np.array([0, 28, 56, 85, 113, 141, 170, 198, 227, 255])
    r_curve = np.array([0, 16, 35, 64, 117, 163, 200, 222, 237, 249])
    g_curve = np.array([0, 24, 49, 98, 141, 174, 201, 223, 239, 255])
    b_curve = np.array([0, 38, 66, 104, 139, 175, 206, 226, 245, 255])
    full_range = np.arange(0, 256)
    b_LUT = np.interp(full_range, x_values, b_curve)
    g_LUT = np.interp(full_range, x_values, g_curve)
    r_LUT = np.interp(full_range, x_values, r_curve)
    b_channel = cv2.LUT(b_channel, b_LUT)
    g_channel = cv2.LUT(g_channel, g_LUT)
    r_channel = cv2.LUT(r_channel, r_LUT)
    img[:, :, 0] = np.uint8(b_channel)
    img[:, :, 1] = np.uint8(g_channel)
    img[:, :, 2] = np.uint8(r_channel)
    return img


def _kelvin(orig):
    img = np.copy(orig)
    b_channel = img[:, :, 0]
    g_channel = img[:, :, 1]
    r_channel = img[:, :, 2]
    orig_r = np.array([0, 60, 110, 150, 235, 255])
    orig_g = np.array([0, 68, 105, 190, 255])
    orig_b = np.array([0, 88, 145, 185, 255])
    r_curve = np.array([0, 102, 185, 220, 245, 245])
    g_curve = np.array([0, 68, 120, 220, 255])
    b_curve = np.array([0, 12, 140, 212, 255])
    full_range = np.arange(0, 256)
    b_LUT = np.interp(full_range, orig_b, b_curve)
    g_LUT = np.interp(full_range, orig_g, g_curve)
    r_LUT = np.interp(full_range, orig_r, r_curve)
    b_channel = cv2.LUT(b_channel, b_LUT)
    g_channel = cv2.LUT(g_channel, g_LUT)
    r_channel = cv2.LUT(r_channel, r_LUT)
    img[:, :, 0] = np.uint8(b_channel)
    img[:, :, 1] = np.uint8(g_channel)
    img[:, :, 2] = np.uint8(r_channel)

    return img


def _adjust_saturation(orig, saturation_scale=1.0):
    img = np.copy(orig)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_img = np.float32(hsv_img)
    H, S, V = cv2.split(hsv_img)
    S = np.clip(S * saturation_scale, 0, 255)
    hsv_img = np.uint8(cv2.merge([H, S, V]))
    im_sat = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return im_sat


def _moon(orig):
    img = np.copy(orig)
    origin = np.array([0, 15, 30, 50, 70, 90, 120, 160, 180, 210, 255])
    _curve = np.array([0, 0, 5, 15, 60, 110, 150, 190, 210, 230, 255])
    full_range = np.arange(0, 256)

    _LUT = np.interp(full_range, origin, _curve)
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_img[:, :, 0] = cv2.LUT(lab_img[:, :, 0], _LUT)
    img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
    img = _adjust_saturation(img, 0.01)
    return img


def clarendon(panel, img_handler, root_handler=None, e=None, init=True):
    if e is not None:
        root_handler.update_func(e.char)
    if init is True:
        output = _clarendon(img_handler.frame)
        img_handler.update_label(panel, output)


def sketch_pencil_using_edge_detection(panel, img_handler, root_handler=None, e=None, init=True):
    if e is not None:
        root_handler.update_func(e.char)
    if init is True:
        output = _sketch_pencil_using_edge_detection(img_handler.frame)
        img_handler.update_label(panel, output)


def xpro2(panel, img_handler, root_handler=None, e=None, init=True):
    if e is not None:
        root_handler.update_func(e.char)
    if init is True:
        output = _xpro2(img_handler.frame)
        img_handler.update_label(panel, output)


def kelvin(panel, img_handler, root_handler=None, e=None, init=True):
    if e is not None:
        root_handler.update_func(e.char)
    if init is True:
        output = _kelvin(img_handler.frame)
        img_handler.update_label(panel, output)


def sketch_pencil_using_blending(panel, img_handler, root_handler=None, e=None, init=True):
    if e is not None:
        root_handler.update_func(e.char)
    if init is True:
        output = _sketch_pencil_using_blending(img_handler.frame)
        img_handler.update_label(panel, output)


def moon(panel, img_handler, root_handler=None, e=None, init=True):
    if e is not None:
        root_handler.update_func(e.char)
    if init is True:
        output = _moon(img_handler.frame)
        img_handler.update_label(panel, output)


def cartoon(panel, img_handler, root_handler=None, e=None, init=True):
    if e is not None:
        root_handler.update_func(e.char)
    if init is True:
        output = _cartoon(img_handler.frame)
        img_handler.update_label(panel, output)


def invert(panel, img_handler, root_handler=None, e=None, init=True):
    if e is not None:
        root_handler.update_func(e.char)
    if init is True:
        output = cv2.bitwise_not(img_handler.frame)
        img_handler.update_label(panel, output)


def black_and_white(panel, img_handler, root_handler=None, e=None, init=True):
    if e is not None:
        root_handler.update_func(e.char)

    if init is True:
        output = cv2.cvtColor(img_handler.frame, cv2.COLOR_BGR2GRAY)
        _, output = cv2.threshold(output, 125, 255, cv2.THRESH_BINARY)
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
        img_handler.update_label(panel, output)


def warming(panel, img_handler, root_handler=None, e=None, init=True):
    if e is not None:
        root_handler.update_func(e.char)
    if init is True:
        output = _warming(img_handler.frame)
        img_handler.update_label(panel, output)


def cooling(panel, img_handler, root_handler=None, e=None, init=True):
    if e is not None:
        root_handler.update_func(e.char)
    if init is True:
        output = _cooling(img_handler.frame)
        img_handler.update_label(panel, output)


def cartoon2(panel, img_handler, root_handler=None, e=None, init=True):
    if e is not None:
        root_handler.update_func(e.char)
    if init is True:
        output = _cartoon2(img_handler.frame)
        img_handler.update_label(panel, output)


def no_filter(panel, img_handler, root_handler=None, e=None, init=True):
    if e is not None:
        root_handler.update_func(e.char)
    if init is True:
        img_handler.update_label(panel, img_handler.frame)
