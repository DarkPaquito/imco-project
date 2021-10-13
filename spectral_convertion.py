import numpy as np
import pandas as pd
import cv2

def compute_xyz(observer, I, S, spacing):
    x, y, z = observer

    N = np.sum(y * I * spacing)

    X = np.sum(x * S * I * spacing) / N
    Y = np.sum(y * S * I * spacing) / N
    Z = np.sum(z * S * I * spacing) / N

    return (X, Y, Z)

def convert_lab(xyz_values, xyz_references):
    xyz_results = xyz_values / xyz_references

    epsilon = 0.008856
    k = 7.787

    for i in range(len(xyz_results)):
        if (xyz_results[i] >  epsilon):
            xyz_results[i] = xyz_results[i] ** (1 / 3)
        else:
            xyz_results[i] = (k * xyz_results[i]) + (16 / 116)

    L = (116 * xyz_results[1]) - 16
    a = 500 * (xyz_results[0] - xyz_results[1])
    b = 200 * (xyz_results[1] - xyz_results[2])
    return L, a, b

def convert_srgb(xyz_values, xyz_references):
    rgb_values = xyz_values / xyz_references

    R = (rgb_values[0] * 3.2406 + rgb_values[1] * -1.5372
            + rgb_values[2] * -0.4986)
    G = (rgb_values[0] * -0.9689 + rgb_values[1] * 1.8758
            + rgb_values[2] * 0.0415)
    B = (rgb_values[0] * 0.0557 + rgb_values[1] * -0.2040
            + rgb_values[2] * 1.0570)
    rgb_results = [R, G, B]

    epsilon = 0.0031308

    for i in range(len(rgb_results)):
        if (rgb_results[i] >  epsilon):
            rgb_results[i] = 1.055 * (rgb_results[i] ** (1 / 2.4)) - 0.055
        else:
            rgb_results[i] = 12.92 * rgb_results[i]

    sR = rgb_results[0] * 255
    sG = rgb_results[1] * 255
    sB = rgb_results[2] * 255
    return sB, sG, sR

def create_table(spectrals, illuminant, observer, xyz_references, spacing):
    results = {}

    for spectral_value in spectrals.index:
        spectral = (spectrals
                    .filter(items=[spectral_value], axis=0)
                    .to_numpy()[0]
                    )

        xyz_values = compute_xyz(observer, illuminant, spectral, spacing)
        lab_values = convert_lab(xyz_values, xyz_references)

        results[spectral_value] = lab_values

    df_results = pd.DataFrame.from_dict(results, orient='index',
            columns=['L', 'a', 'b'])
    return df_results

def create_image(spectrals, illuminant, observer, xyz_references, spacing):
    spectral_values = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
            'K1', 'K2', 'K3', 'K4', 'K5', 'K6', 'K7', 'K8', 'K9']

    image = np.zeros((500, 400, 3), np.uint8)

    x_shift = 0
    y_shift = 0

    for spectral_value in spectral_values:
        spectral = (spectrals
                    .filter(items=[spectral_value], axis=0)
                    .to_numpy()[0]
                    )

        xyz_values = compute_xyz(observer, illuminant, spectral, spacing)
        color = convert_srgb(xyz_values, xyz_references)

        start_point = (0 + x_shift, 0 + y_shift)
        end_point = (100 + x_shift, 100 + y_shift)

        image = cv2.rectangle(image, start_point, end_point, color, -1)

        x_shift += 100

        if x_shift == 400:
            x_shift = 0
            y_shift += 100

    return image

if __name__ == "__main__":
    illuminant_value = input('Specify the illuminant: ')

    if illuminant_value != 'D65' and illuminant_value != 'A':
        print('Unrecognised illuminant value')
        exit(1)

    # Extract required data
    illuminant = np.genfromtxt('spectral_data/{}.csv'.format(illuminant_value),
            delimiter=',', skip_header=1)

    spacing = illuminant[1, 0] - illuminant[0, 0]
    illuminant = illuminant[:, 1]

    observer = np.genfromtxt('spectral_data/CIE1931_CMF.csv', delimiter=',',
            skip_header=1)[:, 1:].T

    spectrals = (
            pd.read_excel('spectral_data/04-b_ColorChart.xlsx',
            sheet_name='Spectral Measurements', header=0, index_col=0)
            )


    xyz_references = (
            pd.read_csv('spectral_data/XYZ_References.csv', delimiter=',',
            skiprows=1, index_col=0)
            .filter(items=[illuminant_value], axis=0)
            .to_numpy()[0]
            )

    # Generate Lab values table
    df_results = create_table(spectrals, illuminant, observer,
            xyz_references, spacing)

    print('Number of values for (a > 20): ',
            len(df_results[df_results['a'] > 20]))

    df_results.to_csv('spectral_data/Lab_Values_{}.csv'.format(illuminant_value),
            index_label='SAMPLE_NAME')

    # Generate sRGB image
    image = create_image(spectrals, illuminant, observer, xyz_references,
            spacing)

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite('spectral_data/sRGB_Colors_{}.tiff'.format(illuminant_value),
            image_bgr)
