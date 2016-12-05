# coding: utf-8
import numpy as np
from Utils import get_matrix_dimensions, gen_standard_output, standard_triangle_check, prod, \
    scale_triangle
from mack_model import estimate_cij as estimate_cij_mack
def estimate_mack_manual(triangle, fjs):
    """
        Estima :math:`C_{i,j}` para :math:`i+j > I` con el modelo manual donde el experto especifica los par?metros de
        desarrollo. Finalmente genera un reporte est?ndar tal y como se detalla en :ref:`el output general <general_output>`.
    :param np.ndarray triangle: Matriz de informaci?n. V?ase la secci?n :ref:`preparaci?n de los datos de entrada <triangle_format>`
        para m?s informaci?n.
    :param np.ndarray fjs: Factores de desarrollo :math:`f_j` preestimados por un experto. La lista fjs\_ debe
        contener las entradas computadas de los par?metros :math:`f_j` desde 0 hasta :math:`j-1`, donde
        se asume que la :math:`j`-?sima entrada de fjs\_ corresponde con :math:`f_j` (:math:`\\text{fjs_[j]} = f_j`).
    :return: Salida est?ndar tal y como se describe en la secci?n :ref:`output general <general_output>`.
    :Posibles errores:
        + La estimaci?n es **NaN**: Las entradas requeridas de la matriz para estimar :math:`f_j` son cero (o casi todas cero).
    """
    I, J = get_matrix_dimensions(triangle)
    scale_factor, triangle = scale_triangle(triangle)
    standard_triangle_check(triangle)
    cijs = np.array([[estimate_cij_mack(triangle, fjs, i_, j_) for j_ in range(J + 1)] for i_ in range(I + 1)])
    deviations_cijs = np.empty((I+1, J+1))
    deviations_cijs[:] = np.NaN
    salida = gen_standard_output(scale_factor*cijs, fjs, deviations_cijs)
    return salida