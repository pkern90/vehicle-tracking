import cv2
import numpy as np
from scipy import signal


def abs_sobel(img_ch, orient='x', sobel_kernel=3):
    """
    Applies the sobel operation on a gray scale image.

    :param img_ch:
    :param orient: 'x' or 'y'
    :param sobel_kernel: an uneven integer
    :return:
    """
    if orient == 'x':
        axis = (1, 0)
    elif orient == 'y':
        axis = (0, 1)
    else:
        raise ValueError('orient has to be "x" or "y" not "%s"' % orient)

    sobel = cv2.Sobel(img_ch, -1, *axis, ksize=sobel_kernel)
    abs_s = np.absolute(sobel)

    return abs_s


def gradient_magnitude(sobel_x, sobel_y):
    """
    Calculates the magnitude of the gradient.
    :param sobel_x:
    :param sobel_y:
    :return:
    """
    abs_grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    return abs_grad_mag.astype(np.uint16)


def gradient_direction(sobel_x, sobel_y):
    """
    Calculates the direction of the gradient. NaN values cause by zero division will be replaced
    by the maximum value (np.pi / 2).
    :param sobel_x:
    :param sobel_y:
    :return:
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        abs_grad_dir = np.absolute(np.arctan(sobel_y / sobel_x))
        abs_grad_dir[np.isnan(abs_grad_dir)] = np.pi / 2

    return abs_grad_dir.astype(np.float32)


def gaussian_blur(img, kernel_size):
    """
    Applies a Gaussian Noise kernel
    :param img:
    :param kernel_size:
    :return:
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def extract_yellow(img):
    """
    Generates an image mask selecting yellow pixels.
    :param img: image with pixels in range 0-255
    :return: Yellow 255 not yellow 0
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (20, 50, 150), (40, 255, 255))

    return mask


def extract_dark(img):
    """
    Generates an image mask selecting dark pixels.
    :param img: image with pixels in range 0-255
    :return: Dark 255 not dark 0
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 0, 0.), (255, 153, 128))
    return mask


def extract_highlights(img, p=99.9):
    """
    Generates an image mask selecting highlights.
    :param p: percentile for highlight selection. default=99.9
    :param img: image with pixels in range 0-255
    :return: Highlight 255 not highlight 0
    """
    p = int(np.percentile(img, p) - 30)
    mask = cv2.inRange(img, p, 255)
    return mask


def binary_noise_reduction(img, thresh):
    """
    Reduces noise of a binary image by applying a filter which counts neighbours with a value
    and only keeping those which are above the threshold.
    :param img: binary image (0 or 1)
    :param thresh: min number of neighbours with value
    :return:
    """
    k = np.array([[1, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    nb_neighbours = cv2.filter2D(img, ddepth=-1, kernel=k)
    img[nb_neighbours < thresh] = 0
    return img


def generate_lane_mask(img, v_cutoff=0):
    """
    Generates a binary mask selecting the lane lines of an street scene image.
    :param img: RGB color image
    :param v_cutoff: vertical cutoff to limit the search area
    :return: binary mask
    """
    window = img[v_cutoff:, :, :]
    yuv = cv2.cvtColor(window, cv2.COLOR_RGB2YUV)
    yuv = 255 - yuv
    hls = cv2.cvtColor(window, cv2.COLOR_RGB2HLS)
    chs = np.stack((yuv[:, :, 1], yuv[:, :, 2], hls[:, :, 2]), axis=2)
    gray = np.mean(chs, 2)

    s_x = abs_sobel(gray, orient='x', sobel_kernel=3)
    s_y = abs_sobel(gray, orient='y', sobel_kernel=3)

    grad_dir = gradient_direction(s_x, s_y)
    grad_mag = gradient_magnitude(s_x, s_y)

    ylw = extract_yellow(window)
    highlights = extract_highlights(window[:, :, 0])

    mask = np.zeros(img.shape[:-1], dtype=np.uint8)

    mask[v_cutoff:, :][((s_x >= 25) & (s_x <= 255) &
                        (s_y >= 25) & (s_y <= 255)) |
                       ((grad_mag >= 30) & (grad_mag <= 512) &
                        (grad_dir >= 0.2) & (grad_dir <= 1.)) |
                       (ylw == 255) |
                       (highlights == 255)] = 1

    mask = binary_noise_reduction(mask, 4)

    return mask


def histogram_lane_detection(img, steps, search_window, h_window):
    """
    Tries to detect lane line pixels by applying a sliding histogram.
    :param img: binary image
    :param steps: steps for the sliding histogram
    :param search_window: Tuple which limits the horizontal search space.
    :param h_window: window size for horizontal histogram smoothing
    :return: x, y of detected pixels
    """
    all_x = []
    all_y = []
    masked_img = img[:, search_window[0]:search_window[1]]
    pixels_per_step = img.shape[0] // steps

    for i in range(steps):
        start = masked_img.shape[0] - (i * pixels_per_step)
        end = start - pixels_per_step
        histogram = np.sum(masked_img[end:start, :], axis=0)
        histogram_smooth = signal.medfilt(histogram, h_window)
        peaks = np.array(signal.find_peaks_cwt(histogram_smooth, np.arange(1, 5)))

        highest_peak = highest_n_peaks(histogram_smooth, peaks, n=1, threshold=5)
        if len(highest_peak) == 1:
            highest_peak = highest_peak[0]
            center = (start + end) // 2
            x, y = get_pixel_in_window(masked_img, highest_peak, center, pixels_per_step)

            all_x.extend(x)
            all_y.extend(y)

    all_x = np.array(all_x) + search_window[0]
    all_y = np.array(all_y)

    return all_x, all_y


def highest_n_peaks(histogram, peaks, n=2, threshold=0):
    """
    Returns the n highest peaks of a histogram above a given threshold.
    :param histogram:
    :param peaks: list of peak indexes
    :param n: number of peaks to select
    :param threshold:
    :return:
    """
    if len(peaks) == 0:
        return []

    peak_list = [(peak, histogram[peak]) for peak in peaks if histogram[peak] > threshold]
    peak_list = sorted(peak_list, key=lambda x: x[1], reverse=True)

    if len(peak_list) == 0:
        return []

    x, y = zip(*peak_list)
    x = list(x)

    if len(peak_list) < n:
        return x

    return x[:n]


def detect_lane_along_poly(img, poly, steps):
    """
    Slides a window along a polynomial an selects all pixels inside.
    :param img: binary image
    :param poly: polynomial to follow
    :param steps: number of steps for the sliding window
    :return: x, y of detected pixels
    """
    pixels_per_step = img.shape[0] // steps
    all_x = []
    all_y = []

    for i in range(steps):
        start = img.shape[0] - (i * pixels_per_step)
        end = start - pixels_per_step

        center = (start + end) // 2
        x = poly(center)

        x, y = get_pixel_in_window(img, x, center, pixels_per_step)

        all_x.extend(x)
        all_y.extend(y)

    return all_x, all_y


def get_pixel_in_window(img, x_center, y_center, size):
    """
    returns selected pixel inside a window.
    :param img: binary image
    :param x_center: x coordinate of the window center
    :param y_center: y coordinate of the window center
    :param size: size of the window in pixel
    :return: x, y of detected pixels
    """
    half_size = size // 2
    window = img[y_center - half_size:y_center + half_size, x_center - half_size:x_center + half_size]

    x, y = (window.T == 1).nonzero()

    x = x + x_center - half_size
    y = y + y_center - half_size

    return x, y


def calculate_lane_area(lanes, area_height, steps):
    """
    Returns a list of pixel coordinates marking the area between two lanes
    :param lanes: Tuple of Lines. Expects the line polynomials to be a function of y.
    :param area_height:
    :param steps:
    :return:
    """
    points_left = np.zeros((steps + 1, 2))
    points_right = np.zeros((steps + 1, 2))

    for i in range(steps + 1):
        pixels_per_step = area_height // steps
        start = area_height - i * pixels_per_step

        points_left[i] = [lanes[0].best_fit_poly(start), start]
        points_right[i] = [lanes[1].best_fit_poly(start), start]

    return np.concatenate((points_left, points_right[::-1]), axis=0)


def are_lanes_plausible(lane_one, lane_two, parallel_thresh=(0.0003, 0.55), dist_thresh=(350, 460)):
    """
    Checks if two lines are plausible lanes by comparing the curvature and distance.
    :param lane_one:
    :param lane_two:
    :param parallel_thresh: Tuple of float values representing the delta threshold for the
    first and second coefficient of the polynomials.
    :param dist_thresh: Tuple of integer values marking the lower and upper threshold
    for the distance between plausible lanes.
    :return:
    """
    is_parallel = lane_one.is_current_fit_parallel(lane_two, threshold=parallel_thresh)
    dist = lane_one.get_current_fit_distance(lane_two)
    is_plausible_dist = dist_thresh[0] < dist < dist_thresh[1]

    return is_parallel & is_plausible_dist


def draw_poly(img, poly, steps, color, thickness=10, dashed=False):
    """
    Draws a polynomial onto an image.
    :param img:
    :param poly:
    :param steps:
    :param color:
    :param thickness:
    :param dashed:
    :return:
    """
    img_height = img.shape[0]
    pixels_per_step = img_height // steps

    for i in range(steps):
        start = i * pixels_per_step
        end = start + pixels_per_step

        start_point = (int(poly(start)), start)
        end_point = (int(poly(end)), end)

        if dashed is False or i % 2 == 1:
            img = cv2.line(img, end_point, start_point, color, thickness)

    return img


def draw_poly_arr(img, poly, steps, color, thickness=10, dashed=False, tip_length=1):
    """
    Draws a polynomial onto an image using arrows.
    :param img:
    :param poly:
    :param steps:
    :param color:
    :param thickness:
    :param dashed:
    :param tip_length:
    :return:
    """
    img_height = img.shape[0]
    pixels_per_step = img_height // steps

    for i in range(steps):
        start = i * pixels_per_step
        end = start + pixels_per_step

        start_point = (int(poly(start)), start)
        end_point = (int(poly(end)), end)

        if dashed is False or i % 2 == 1:
            img = cv2.arrowedLine(img, end_point, start_point, color, thickness, tipLength=tip_length)

    return img


def outlier_removal(x, y, q=5):
    """
    Removes horizontal outliers based on a given percentile.
    :param x: x coordinates of pixels
    :param y: y coordinates of pixels
    :param q: percentile
    :return: cleaned coordinates (x, y)
    """
    if len(x) == 0 or len(y) == 0:
        return x, y

    x = np.array(x)
    y = np.array(y)

    lower_bound = np.percentile(x, q)
    upper_bound = np.percentile(x, 100 - q)
    selection = (x >= lower_bound) & (x <= upper_bound)
    return x[selection], y[selection]
