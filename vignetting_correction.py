import cv2
import numpy as np

# TODO Opti tout ça
def compute_H(a, b, c, img_gray):
    rows, cols = img_gray.shape

    img_gray_float = np.zeros((rows, cols), dtype=float)

    c_x, c_y = cols / 2.0, rows / 2.0
    d = np.sqrt(c_x ** 2 + c_y ** 2)

    for row in range(rows):
        for col in range(cols):
            r = np.sqrt((row - c_y) ** 2 + (col - c_x) ** 2) / d
            g = 1 + a * (r ** 2) + b * (r ** 4) + c * (r ** 6)

            img_gray_float[row, col] = img_gray[row, col] * g

    img_log = np.zeros((rows, cols), dtype=float)

    for row in range(rows):
        for col in range(cols):
            img_log[row, col] = 255 * (np.log(1 + img_gray_float[row, col]) / 8)

    histo = np.zeros(256, dtype=float)

    for row in range(rows):
        for col in range(cols):
            k_d = int(np.floor(img_log[row, col]))
            k_u = int(np.ceil(img_log[row, col]))

            histo[k_d] += (1 + k_d - img_log[row, col])
            histo[k_u] += (k_u - img_log[row, col])

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
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    a, b, c = 0., 0., 0.
    a_min, b_min, c_min = 0., 0., 0.
    delta = 8

    H_min = compute_H(a, b, c, img_gray)

    while delta > (1 / 256):
        print("Coucou")
        a_tmp = a + delta

        if check(a_tmp, b, c):
            H = compute_H(a_tmp, b, c, img_gray)

            if H_min > H:
                a_min = a_tmp
                b_min = b
                c_min = c

                H_min = H

        a_tmp = a - delta

        if check(a_tmp, b, c):
            H = compute_H(a_tmp, b, c, img_gray)

            if H_min > H:
                a_min = a_tmp
                b_min = b
                c_min = c

                H_min = H

        b_tmp = b - delta

        if check(a, b_tmp, c):
            H = compute_H(a, b_tmp, c, img_gray)

            if H_min > H:
                a_min = a
                b_min = b_tmp
                c_min = c

                H_min = H

        c_tmp = c + delta

        if check(a, b, c_tmp):
            H = compute_H(a, b, c_tmp, img_gray)

            if H_min > H:
                a_min = a
                b_min = b
                c_min = c_tmp

                H_min = H

        c_tmp = c - delta

        if check(a, b, c_tmp):
            H = compute_H(a, b, c_tmp, img_gray)

            if H_min > H:
                a_min = a
                b_min = b
                c_min = c_tmp

                H_min = H

        delta /= 2.0

    print(a_min, b_min, c_min)

    img_result = np.zeros(img_gray.shape, dtype=int)
    rows, cols = img_gray.shape

    c_x, c_y = cols / 2.0, rows / 2.0
    d = np.sqrt(c_x ** 2 + c_y ** 2)

    for row in range(rows):
        for col in range(cols):
            r = np.sqrt((row - c_y) ** 2 + (col - c_x) ** 2) / d
            g = 1 + a_min * (r ** 2) + b_min * (r ** 4) + c_min * (r ** 6)

            img_result[row, col] = img_gray[row, col] * g
            if img_result[row, col] > 255:
                img_result[row, col] = 255
            if img_result[row, col] < 0:
                img_result[row, col] = 0

    return img_result

if __name__ == '__main__':
    img = cv2.imread("test_vignetting_2.jpg")

    img_result = vignetting_correction(img)

    cv2.imwrite("image_processed.jpg", img_result)
