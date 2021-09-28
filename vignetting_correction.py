import cv2
import numpy as np

def compute_entropy(a, b, c, img_l):
    rows, cols = img_l.shape

    histo = np.zeros(256, dtype=float)

    c_x, c_y = cols / 2.0, rows / 2.0
    d = np.sqrt(c_x ** 2 + c_y ** 2)

    for row in range(rows):
        for col in range(cols):
            r = np.sqrt((row - c_y) ** 2 + (col - c_x) ** 2) / d
            g = 1 + a * (r ** 2) + b * (r ** 4) + c * (r ** 6)

            gray_float = img_l[row, col] * g

            log = 255 * (np.log(1 + gray_float) / 8)

            k_d = int(np.floor(log))
            k_u = int(np.ceil(log))

            histo[k_d] += (1 + k_d - log)
            histo[k_u] += (k_u - log)

    tmp_histo = np.zeros(264, dtype=float)
    tmp_histo[0], tmp_histo[1], tmp_histo[2], tmp_histo[3] = (
            histo[4], histo[3], histo[2], histo[1]
            )
    tmp_histo[260], tmp_histo[261], tmp_histo[262], tmp_histo[263] = (
            histo[254], histo[253], histo[252], histo[251]
            )
    tmp_histo[4:260] = histo

    for i in range(256):
        histo[i] = (tmp_histo[i] + 2 * tmp_histo[i + 1] + 3 * tmp_histo[i + 2]
                + 4 * tmp_histo[i + 3] + 5 * tmp_histo[i + 4]
                + 4 * tmp_histo[i + 5] + 3 * tmp_histo[i + 6]
                + 2 * tmp_histo[i + 7]) + tmp_histo[i + 8] / 25.0

    sum_histo = np.sum(histo)

    H = 0
    for i in range(256):
        pk = histo[i] / sum_histo
        if pk != 0:
            H += pk * np.log(pk)

    return -H

def check(a, b, c):
    if a == 0 and b == 0 and c == 0:
        return False

    if a > 0 and b == 0 and c == 0:
        return True

    if a >= 0 and b > 0 and c == 0:
        return True

    if c == 0 and b < 0 and -a <= 2 * b:
        return True

    if c > 0 and b * b < 3 * a * c:
        return True

    if c > 0 and b * b == 3 * a * c and b >= 0:
        return True

    if c > 0 and b * b == 3 * a * c and -b >= 3 * c:
        return True

    q_p = (-2 * b + np.sqrt(4 * b * b - 12 * a * c)) / (6 * c)
    if c > 0 and b * b > 3 * a * c and q_p <= 0:
        return True

    q_d = (-2 * b - np.sqrt(4 * b * b - 12 * a * c)) / (6 * c)
    if c > 0 and b * b > 3 * a * c and q_d >= 1:
        return True

    if c < 0 and b * b > 3 * a * c and q_p >=1 and q_d<=0:
        return True
    return False


def vignetting_correction(img):
    img_HSL = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    img_l = img_HSL[:, :, 1]

    a, b, c = 0., 0., 0.
    a_min, b_min, c_min = 0., 0., 0.
    delta = 8.0

    H_min = compute_entropy(a, b, c, img_l)

    while delta > (1 / 256):
        print("delta:", delta)

        a_tmp = a + delta
        if check(a_tmp, b, c):
            H = compute_entropy(a_tmp, b, c, img_l)

            if H_min > H:
                a_min = a_tmp
                b_min = b
                c_min = c

                H_min = H

        a_tmp = a - delta
        if check(a_tmp, b, c):
            H = compute_entropy(a_tmp, b, c, img_l)

            if H_min > H:
                a_min = a_tmp
                b_min = b
                c_min = c

                H_min = H

        b_tmp = b + delta
        if check(a, b_tmp, c):
            H = compute_entropy(a, b_tmp, c, img_l)

            if H_min > H:
                a_min = a
                b_min = b_tmp
                c_min = c

                H_min = H

        b_tmp = b - delta
        if check(a, b_tmp, c):
            H = compute_entropy(a, b_tmp, c, img_l)

            if H_min > H:
                a_min = a
                b_min = b_tmp
                c_min = c

                H_min = H

        c_tmp = c + delta
        if check(a, b, c_tmp):
            H = compute_entropy(a, b, c_tmp, img_l)

            if H_min > H:
                a_min = a
                b_min = b
                c_min = c_tmp

                H_min = H

        c_tmp = c - delta
        if check(a, b, c_tmp):
            H = compute_entropy(a, b, c_tmp, img_l)

            if H_min > H:
                a_min = a
                b_min = b
                c_min = c_tmp

                H_min = H

        delta /= 2.0

    print(a_min, b_min, c_min)

    rows, cols = img_l.shape

    c_x, c_y = cols / 2.0, rows / 2.0
    d = np.sqrt(c_x ** 2 + c_y ** 2)

    for row in range(rows):
        for col in range(cols):
            r = np.sqrt((row - c_y) ** 2 + (col - c_x) ** 2) / d
            g = 1 + a_min * (r ** 2) + b_min * (r ** 4) + c_min * (r ** 6)

            img_HSL[row, col, 1] = np.clip(img_HSL[row, col, 1] * g, 0, 255)

    return cv2.cvtColor(img_HSL, cv2.COLOR_HLS2BGR)

if __name__ == '__main__':
    img = cv2.imread("image_test/test_vignetting_2.jpg")

    img_result = vignetting_correction(img)

    cv2.imwrite("image_test/image_processed_2_HSL.jpg", img_result)
