# coding: utf-8
from ferguson_model import estimate_cij as estimate_cij_ferguson, fj, \
    gen_standard_output, sigma_square, gamma, calc_mu, mij, d, phi, \
    single_payments_from_triangle, h1, h2, h3, fisher_matrix, g_matrix, \
    estimation_error as estimation_error_ferguson, cij_deviation as cij_deviation_ferguson
from Utils import get_matrix_dimensions, prod, xijs_from_cijs, msep, scale_triangle
from mack_model import cij_deviation as cij_deviation_mack, sj, \
    estimation_error as estimation_error_mack
import numpy as np
def estimate_cij(triangle, fjs, cijs_ferguson, i, j):
    """
        Estima :math:`C_{i, j}` mediante el modelo Hovinen entendido como
        el monto por concepto de incurridos en el período :math:`i` reportados
        hasta :math:`j` períodos después.
    :param np.ndarray triangle: Matriz de información. Véase la sección :ref:`preparación de los datos de entrada <triangle_format>`
        para más información.
    :param fjs: Array de tamaño :math:`J` donde el :math:`j`-ésimo elemento corresponde con el valor
        estimado del factor de desarrollo :math:`f_j`.
    :param cijs_ferguson: Matriz de tamaño :math:`I \\times J` donde la entrada :math:`i, j` corresponde
        con la estimación de :math:`C_{i, j}` por parte del modelo Ferguson.
    :param int i: Período en el que sucedió el siniestro. Toma valores desde 0 hasta :math:`I`.
    :param int j: Cantidad de períodos transcurridos después de sucedido el siniestro hasta que fue reportado.
        Toma valores desde 0 hasta :math:`J`.
    :return: Si :math:`i+j \\leq I`, devuelve la posición :math:`i, j` de la matriz triangle. Si no, retorna
        :math:`C_{i, j}^{Hov} = C_{i, I-i} + (1-\\beta_{I-i})C_{i, j}^{Fer}`
        donde
        :math:`\\beta_j=\\prod_{k=j}^{J-1}\\frac{1}{f_k}`,
        :math:`C_{i, j}^{Hov}` es la estimación mediante el modelo Hovinen de :math:`C_{i, j}`,
        y :math:`C_{i, j}^{Fer}` es la estimación mediante el modelo Ferguson de :math:`C_{i, j}`.
    :Posibles errores:
        + **AssertionError**:
            + El parámetro :math:`i` no es un valor entero entre 0 e I.
            + El parámetro :math:`j` no es un valor entero entre 0 y J.
            + No se han calculado los suficientes valores de :math:`f_j`.
    """
    I, J = get_matrix_dimensions(triangle)
    assert i in range(I+1), "Período de incurrido a estimar inválido"
    assert j in range(J+1), "Período de reportado a estimar inválido"
    assert len(fjs)+1 > j, "Se requieren más fj's calculados para estimar cij"
    if i+j <= I:
        return triangle[i, j]
    beta = prod(1.0/fjs[I - i: J])
    return triangle[i, I-i] + (1-beta)*cijs_ferguson[i, j]
def cij_deviation(cijs_deviations_mack, fjs, i, j):
    """
        Desviación de :math:`C_i, j` con el modelo Hovinen.
    :param np.ndarray cijs_deviations_mack: Matriz donde la :math:`i, j`-ésima entrada corresponde con
        la desviación de :math:`C_{i, j}` según Mack.
    :param np.array fjs: Lista donde la :math:`j`-ésima entrada corresponde con el
        factor de desarrollo :math:`f_j`.
    :param i: Período de ocurrencia en el que se va a calcular la desviación. Puede tomar
        valores :math:`i=0, \\cdots, I`
    :param j: Período de reporte en el cuál se va a calcular la desviación. Puede tomar
        valores :math:`i=0, \\cdots, J`.
    :return: :math:`(1-\\beta_{I-i})\\beta_{I-i}C_{i, j}^{CL}`
        donde :math:`\\beta_j=\\prod_{k=j}^{J-1}\\frac{1}{f_k}`.
    """
    I, J = get_matrix_dimensions(cijs_deviations_mack)
    if i+j <= I:
        return 0.0
    beta = prod(1.0/fjs[I - i: J])
    return ((1-beta) * beta)*cijs_deviations_mack[i, j]
def msep_hovinen(fjs, msep_cijs_ferguson, msep_cijs_mack, deviations_cij, i, I, J):
    """
        Error de estimación de Hovinen.
    :param fjs: Factores de desarrollo. Es una lista con :math:`J` elementos donde el
        :math:`j`-ésimo elemento corresponde con :math:`f_j`.
    :param msep_cijs_ferguson: Array de tamaño :math:`J` donde la :math:`i`-ésima posición corresponde
        con el msep calculado para :math:`C_{I-J + i, J}` según el modelo Fergsuon.
    :param msep_cijs_mack: Array de tamaño :math:`J` donde la :math:`i`-ésima posición corresponde
        con el msep calculado para :math:`C_{I-J + i, J}` según el modelo Mack.
    :param i: Período reportado del suceso.
    :param I: Período hasta el que se tiene información.
    :param J: Período hasta el que se tiene información sobre reportados.
    :return:
    """
    if i == 0:
        return 0.0
    beta = prod(1.0/fjs[I - i: J])
    t = (1 - beta) * 1.0 / ((1.0 / beta) * (msep_cijs_ferguson[i] * 1.0 / msep_cijs_mack[i]) - 1)
    msep_teorico = msep_cijs_mack[i] * beta * (1 - beta) * (beta + 1.0 / (1 - beta) + ((1 - beta) ** 2) * 1.0 / t)
    result = max(msep_teorico, deviations_cij[i, -1]**2)
    return result
def compute_ferguson_msep(triangle, fjs, nus, variances):
    """
        Calcula el msep (mean square error prediction) mediante el modelo Ferguson.
    :param np.ndarray triangle: Matriz de información. Véase la sección :ref:`preparación de los datos de entrada <triangle_format>`
        para más información.
    :param fjs: Factores de desarrollo. Es una lista con :math:`J` elementos donde el
        :math:`j`-ésimo elemento corresponde con :math:`f_j`.
    :param np.array nus:
        Estimación *a priori* del parámetro :math:`\\nu_i`. Es una lista de tamaño :math:`J`, donde la :math:`i`-ésima entrada
        es la estimación *a priori* por parte de un experto de :math:`C_{i, J}` (pago de siniestros incurridos en el período :math:`i`
        reportados en el período :math:`J`) para :math:`i=I-J, I-J+1, \\cdots, I` .
    """
    I, J = get_matrix_dimensions(triangle)
    single_payments_matrix = single_payments_from_triangle(triangle)
    gammas = np.array([gamma(fjs, j, J) for j in range(J + 1)])
    calc_mus = np.array([calc_mu(triangle, fjs, i) for i in range(I + 1)])
    mijs = np.array([[mij(gammas, calc_mus, i_, j_) for j_ in range(J + 1)] for i_ in range(I + 1)])
    d_ = d(I)
    phi_ = phi(single_payments_matrix, d_, mijs)
    h1s = np.array([h1(phi_, gammas, calc_mus, i, I) for i in range(I + 1)])
    h2s = np.array([h2(phi_, gammas, calc_mus, j, I) for j in range(I + 1)])
    param_h3_comb = [(j, l) for j in range(J) for l in range(J) if j != l]
    h3s = np.array([h3(phi_, gammas, calc_mus, j_, l, I) for j_, l in param_h3_comb])
    fisher_matrix_ = fisher_matrix(phi_, h1s, h2s, h3s, I)
    fisher_matrix_inverse = np.linalg.inv(fisher_matrix_)
    g_matrix_ = g_matrix(fisher_matrix_inverse, I)
    est_errors_ferguson = np.array([estimation_error_ferguson(gammas, g_matrix_, i, J, I, nus, variances) for i in range(I + 1)])
    deviations_cijs_ferguson = np.array([[cij_deviation_ferguson(phi_, gammas, i, j, I, nus) for j in range(J+1)] for i in range(I + 1)])
    msep_ferguson = msep(est_errors_ferguson, deviations_cijs_ferguson[:, -1])
    return msep_ferguson
def estimate_hovinen(triangle, nus, variances):
    """
        Estimación mediante el modelo Hovinen de :math:`C_{i, j}`.
    :param np.ndarray triangle: Matriz de información. Véase la sección :ref:`preparación de los datos de entrada <triangle_format>`
        para más información.
    :param np.array nus:
        Estimación *a priori* del parámetro :math:`\\nu_i`. Es una lista de tamaño :math:`J`, donde la :math:`i`-ésima entrada
        es la estimación *a priori* por parte de un experto de :math:`C_{i, J}` (pago de siniestros incurridos en el período :math:`i`
        reportados en el período :math:`J`) para :math:`i=I-J, I-J+1, \\cdots, I` .
    :return: Salida estándar tal y como se describe en la sección :ref:`general_output`.
    :Posibles errores:
        + La estimación es **NaN**: Las entradas requeridas de la matriz para estimar :math:`f_j` son cero (o casi todas cero).
    """
    nus = np.array(nus, dtype=float)
    variances = np.array(variances, dtype=float)
    I, J = get_matrix_dimensions(triangle)
    scale_factor, triangle = scale_triangle(triangle)
    nus *= 1.0/scale_factor
    variances /= 1.0*scale_factor**2
    assert I == J, "Aún no se soportan matrices rectangulares."
    assert len(nus) >= I, "Insuficientes estimaciones a priori para llevar a cabo la estimación con Ferguson"
    fjs = np.array([fj(triangle, j) for j in range(J)])
    cijs_ferguson = np.array([[estimate_cij_ferguson(triangle, fjs, i_, j_, nus) for j_ in range(J + 1)] for i_ in range(I + 1)])
    cijs = np.array([[estimate_cij(triangle, fjs, cijs_ferguson, i, j) for j in range(J + 1)] for i in range(I + 1)])
    sigma_squares = np.array([sigma_square(triangle, fjs, j) for j in range(J)])
    sjs = np.array([sj(triangle, j) for j in range(J)])
    cijs_deviations_mack = np.array([[cij_deviation_mack(cijs, fjs, sigma_squares, i, j) for j in range(J + 1)] for i in range(I + 1)])
    est_errors_mack = np.array([estimation_error_mack(triangle, fjs, sjs, sigma_squares, i, J) for i in range(I + 1)])
    deviations_cijs = np.array([[cij_deviation(cijs_deviations_mack, fjs, i, j) for j in range(J+1)] for i in range(I+1)])
    msep_mack = msep(est_errors_mack, cijs_deviations_mack[:, -1])
    msep_ferguson = compute_ferguson_msep(triangle, fjs, nus, variances)
    hovinen_msep = np.array([msep_hovinen(fjs, msep_ferguson, msep_mack, deviations_cijs, i, I, J) for i in range(I + 1)])
    estimation_errors_last = np.sqrt(hovinen_msep - deviations_cijs[:, -1]**2)
    estimation_errors = np.empty((I + 1, J + 1,))
    estimation_errors[:, :-1] = np.nan
    estimation_errors[:, -1] = estimation_errors_last
    return gen_standard_output(scale_factor * cijs, fjs, scale_factor * deviations_cijs)