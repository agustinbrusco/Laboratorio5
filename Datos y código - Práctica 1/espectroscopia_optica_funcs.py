from typing import Tuple, Callable
import numpy as np
import scipy.constants as cte
from scipy.signal import find_peaks
import cv2 as cv


def balmer_wavelength(n: int) -> float:  # m
    '''Calcula la longitud de onda asociada a la linea de Balmer debido a la
    transición de un electrón desde el nivel n hasta el nive 2.
    '''
    # R_H según: https://es.wikipedia.org/wiki/Constante_de_Rydberg
    Ryd_hidrogeno = cte.Rydberg/(1 + cte.m_e/(cte.m_p + cte.m_n))  # m⁻¹
    return 1/(Ryd_hidrogeno*(1/4 - 1/n**2))  # m


def wavelength_to_RGB(wavelength: float) -> Tuple[float, float, float, ]:
    '''Obtenida de: http://www.noah.org/wiki/Wavelength_to_RGB_in_Python
    Convierte una determinada longitud de onda de luz en un valor de color
    RGB aproximado. La longitud de onda debe darse en nanómetros en el rango
    de 380 nm a 750 nm (789 THz a 400 THz). Para valores fuera de este rango,
    devuelve la tupla correspondiente al color blanco.

    Basado en el código de Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    '''
    gamma = 0.8
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7*(wavelength - 380)/(440 - 380)
        R = ((-(wavelength - 440)/(440 - 380))*attenuation)**gamma
        G = 0.0
        B = attenuation**gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440)/(490 - 440))**gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = ((510 - wavelength)/(510 - 490))**gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510)/(580 - 510))**gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = ((645 - wavelength)/(645 - 580))**gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7*(750 - wavelength)/(750 - 645)
        R = attenuation**gamma
        G = 0.0
        B = 0.0
    else:
        R = 1.0
        G = 1.0
        B = 1.0
    return (R, G, B,)


def get_linear_transformation(x0: float, x1: float,
                              y0: float, y1: float) -> Callable:
    m = (y1 - y0)/(x1 - x0)
    b = y0 - m*x0
    return lambda x: m*x + b


def get_column_intensity(file: str) -> np.ndarray:
    img = cv.imread(file,)
    grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return np.sum(grayscale, axis=0)/grayscale.shape[0]


def get_wavelength_from_He(He_file: str) -> np.ndarray:
    '''Dada una imagen tomada del monocromador en la que se observan las
    lineas del helio, compara los máximos de intensidad medidos con los
    obtenidos del espectrómetro CCS200 e interpola (y extrapola) linealmente
    para devolver un array de longitudes de onda en nanometros. El valor en el
    i-ésimo elemento del array se corresponde con la longitud de onda de la luz
    que es medida en la i-ésima columna de pixeles en una imagen tomada con
    el monocromador y la cámara colocados en la misma posición.
    '''
    intensidad = get_column_intensity(He_file)  # a.u.
    peaks = find_peaks(intensidad,  # Indices de los picos de intensidad
                       prominence=10,
                       distance=20)
    peak_vals = intensidad[peaks[0]]  # a.u. : Intensidad medida en los máximos
    order_max_max = np.argmax(peak_vals)  # Índice del máximo de intensidad
    if order_max_max == 0:  # Considero los primeros dos máximos detectados
        x0, x1 = peaks[0][:2]
        pixel_to_wavelen = get_linear_transformation(x0, x1,
                                                     y0=588.87, y1=669.07)
    else:  # Considero el mayor máximo y el anterior
        x0, x1 = peaks[0][[order_max_max-1, order_max_max]]
        pixel_to_wavelen = get_linear_transformation(x0, x1,
                                                     y0=502.60, y1=588.87)
    return pixel_to_wavelen(np.arange(intensidad.size))  # nm
