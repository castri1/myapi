# coding: utf-8
import numpy as np
from Utils import get_matrix_dimensions, gen_standard_output, fj, vco_matrix, \
    xijs_from_cijs, standard_triangle_check, prod, prediction_error, sigma_square, \
    scale_triangle
def estimate_cij(triangle, fjs_, i, j):
    """
        Estimador de :math:`C_{i, j}` interpretado como el pago acumulado de siniestros incurridos en el período :math:`i`
        reportados hasta :math:`j` períodos después.
    :param np.ndarray fjs_: Estimación de los factores de desarrollo :math:`f_j`. La lista fjs\_ debe
        contener las entradas computadas de los parámetros :math:`f_j` desde 0 hasta :math:`j-1`, donde
        se asume que la :math:`j`-ésima entrada de fjs\_ corresponde con :math:`f_j` (:math:`\\text{fjs_[j]} = f_j`).
    :param np.ndarray triangle: Matriz de información. Véase la sección :ref:`preparación de los datos de entrada <triangle_format>`
        para más información.
    :param int i: Período en el que sucedió el siniestro. Toma valores desde 0 hasta :math:`I`.
    :param int j: Cantidad de períodos transcurridos después de sucedido el siniestro hasta que fue reportado.
        Toma valores desde 0 hasta :math:`J`.
    :return: Cuando :math:`i+j > I`, devuelve
        :math:`C_{i,j} = C_{i, I-i}f_{I-i}\\cdots f_{j-1}`.
        Cuando :math:`i+j \\leq I`, devuelve la posición :math:`i, j` de la matriz triangle.
    :Posibles errores:
        + **AssertionError**:
            + El parámetro :math:`i` no es un valor entero entre 0 e I.
            + El parámetro :math:`j` no es un valor entero entre 0 y J.
            + No se han calculado los suficientes valores de :math:`f_j`.
        + La estimación es **NaN**: Las entradas requeridas de la matriz para estimar :math:`f_j` son cero (o casi todas cero).
    """
    I, J = get_matrix_dimensions(triangle)
    assert i in range(I+1), "Período de incurrido a estimar inválido"
    assert j in range(J+1), "Período de reportado a estimar inválido"
    assert len(fjs_)+1 > j, "Se requieren más fj's calculados para estimar cij"
    if i+j <= I:
        return triangle[i, j]
    fjs = fjs_[I - i: j]
    cij = triangle[i, I - i]
    return prod(fjs)*cij
def sj(triangle, j):
    """
        Estimador del parámetro :math:`s_j` requerido para calcular el error de estimación.
    :param np.ndarray triangle: Matriz de información. Véase la sección :ref:`preparación de los datos de entrada <triangle_format>`
        para más información.
    :param int j: Entero con valor entre 0 y :math:`J-1` (incluyendo) donde :math:`J` es el número de columnas
        de la matriz triangle.
    :return: Para :math:`j=0,1,\cdots, J`, donde :math:`J` el número de columnas de la matriz triangle, devuelve
        :math:`s_j = \\sum_{k=0}^{I-j-1}C_{k, j}`.
    :Posibles errores:
        + **AssertionError**:
            + El parámetro :math:`j` no es un valor entero entre 0 y J
    """
    I, J = get_matrix_dimensions(triangle)
    assert j in range(J+1), "j debe ser un valor entero entre 0 y J, donde J es el número de columnas" \
                            " de la matriz trángulo menos 1."
    # Todo: Si toda la columna es cero, hacer value error, hacerlo desde la función main.
    return sum(triangle[:I - j, j])
def xijs_deviation(cijs, sigma_squares, fjs, deviations_cijs, i, j):
    """
        Estima la desviación de :math:`X_{i,j}`.
    :param np.ndarray cijs: Matriz del mismo tamaño que el :ref:`triángulo inicial <triangle_format>`
        pero con los valores restantes ya estimados, y donde cada entrada corresponde con :math:`C_{i, j}`.
        Como :math:`C_{i, j}` para :math:`i+j \leq I` son valores ya dados, estas entradas deben
        ser iguales en ambas matrices (cijs y el :ref:`triángulo inicial <triangle_format>` ).
    :param np.ndarray sigma_squares: Lista de tamaño :math:`J` de parámetros :math:`\\sigma_j`
        para :math:`j = 0, 1\\cdots J-1` ya estimados.
    :param np.ndarray fjs: Estimación de :math:`f_j`. La lista fjs debe
        contener las entradas computadas de los parámetros :math:`f_j` desde 0 hasta j-1, donde
        se asume que la :math:`j`-ésima entrada de fjs corresponde con :math:`f_j` (:math:`\\text{fjs[j]} = f_j`).
    :param np.ndarray deviations_cijs: Matriz del mismo tamaño que la matriz cijs donde la entrada :math:`i, j` corresponde
        con la desviación del parámetro :math:`C_{i, j}`
    :param int i: Índice que indica el período de incurridos al cual se le va a estimar la desviación de :math:`x_{i, j}`.
        Toma valores :math:`i=0, \cdots I`.
    :param int j: Índice que indica el período de reportados al cual se le va a estimar la desviación de :math:`x_{i, j}`.
        Toma valores :math:`j=0, \cdots J`.
    :return: Cálculo de
        :math:`\\text{desviación de } X_{i, j} = \\sqrt{\\sigma_{j-1}^2 C_{i, j-1} + (f_{j-1}-1)\\text{var}(C_{i, j-1})}`
    """
    I, J = get_matrix_dimensions(cijs)
    if i+j > I:
        return np.sqrt(sigma_squares[j-1] * cijs[i, j-1] + ((fjs[j-1]-1) * deviations_cijs[i, j - 1]) ** 2)
    else:
        return 0
def cij_deviation(cijs_, fjs_, sigma_squares_, i, j):
    """
        Estimador de la desviación de :math:`C_{i, j}` para el modelo Mack, identificado tambien como el cuadrado del error del modelo.
    :param np.ndarray cijs_: Matriz (numpy array) de tamaño :math:`I \\times J`
        donde la entrada :math:`i, j` con :math:`i+j \\leq I` es la entrada :math:`i, j` de
        la :ref:`matriz de datos <triangle_format>`. Para :math:`i+j > I` la entrada :math:`i, j` es el valor estimado
        de :math:`C_{i, j}` con el modelo Mack.
    :param np.ndarray fjs_: Estimación de los factores de desarrollo :math:`f_j`. La lista fjs\_ debe
        contener las entradas computadas de los parámetros :math:`f_j` desde 0 hasta j-1, donde
        se asume que la :math:`j`-ésima entrada de fjs\_ corresponde con :math:`f_j` (:math:`\\text{fjs_[j]} = f_j`).
    :param sigma_squares_: Estimación de :math:`\\sigma_j`. La lista sigma_squares\_ debe
        contener las entradas computadas de los parámetros :math:`\sigma_j` desde 0 hasta j-1, donde
        se asume que la :math:`j`-ésima entrada de sigma_squares\_ corresponde con :math:`\sigma_j` (:math:`\\text{sigma_squares_[j]} = f_j`).
    :param int i: Período en el que sucedió el siniestro. Toma valores desde 0 hasta :math:`I`.
    :param int j: Cantidad de períodos transcurridos después de sucedido el siniestro hasta que fue reportado.
        Toma valores desde 0 hasta :math:`J`.
    :return:  Cuando :math:`i+j > I`, devuelve:
        :math:`\\sqrt{\\text{variance}} = \\sigma = \\sqrt{C_{i, j}^2\\sum_{k=I-i}^{I-1}\\frac{{\\sigma_k^2}/{f_k^2}}{C_{i, k}}}`
        donde :math:`C_{i, j}` es la entrada :math:`i, j` de la matriz cijs\_. Cuando :math:`i+j \\leq I`, devuelve 0.
    :Posibles errores:
        + **AssertionError**:
            + El parámetro :math:`i` no es un valor entero entre 0 e I
            + El parámetro :math:`j` no es un valor entero entre 0 y J.
            + No se han calculado los suficientes valores de :math:`f_j`.
    :return: Cálculo de
        :math:`\\text{desviación de } C_{i, j} = \\sqrt{C_{i, j}^2\\sum_{k=I-i}^{j}\\frac{\\sigma_k^2/f_k^2}{C_{i, k}}}`
    """
    I, J = get_matrix_dimensions(cijs_)
    assert i in range(I+1), "Período de incurrido a estimar inválido"
    assert j in range(cijs_.shape[1]), "Período de reportado a estimar inválido"
    assert len(fjs_)+1 > j, "Se requieren más fj's calculados para estimar cij"
    if i+j <= I:
        return 0.0
    cij = cijs_[i, j]
    cij_l = cijs_[i, I - i: j]
    fj_l = fjs_[I - i: j]
    sigma_sq = sigma_squares_[I - i: j]
    return np.sqrt((cij ** 2) * sum((sigma_sq * 1.0 / fj_l ** 2) * 1.0 / cij_l))
def estimation_error(triangle, fjs_, sjs_, sigma_squares_, i, j):
    """
        Error de estimación para el modelo Mack.
    :param np.ndarray triangle: Matriz de información. Véase la sección :ref:`preparación de los datos de entrada <triangle_format>`
        para más información.
    :param np.ndarray fjs_: Estimación de :math:`f_j`. La lista fjs\_ debe
        contener las entradas computadas de los parámetros :math:`f_j` desde 0 hasta j-1, donde
        se asume que la :math:`j`-ésima entrada de fjs\_ corresponde con :math:`f_j` (:math:`\\text{fjs_[j]} = f_j`).
    :param np.ndarray sjs_: Estimación de :math:`s_j`. La lista sjs\_ debe
        contener las entradas computadas de los parámetros :math:`s_j` desde 0 hasta j-1, donde
        se asume que la :math:`j`-ésima entrada de sjs\_ corresponde con :math:`s_j` (:math:`\\text{sjs_[j]} = s_j`).
    :param sigma_squares_: Estimación de :math:`\sigma_j`. La lista sigma_squares\_ debe
        contener las entradas computadas de los parámetros :math:`\sigma_j` desde 0 hasta j-1, donde
        se asume que la :math:`j`-ésima entrada de sigma_squares\_ corresponde con :math:`\sigma_j` (:math:`\\text{sigma_squares_[j]} = f_j`).
    :param int i: Período en el que sucedió el siniestro. Toma valores desde 0 hasta :math:`I`.
    :param int j: Cantidad de períodos transcurridos después de sucedido el siniestro hasta que fue reportado.
        Toma valores desde 0 hasta :math:`J`.
    :return:  Cuando :math:`i+j > I`, devuelve:
        :math:`\\text{error estimación}_{i, j} = C_{i, I-i}^2\\left( \\prod_{k=I-i}^{j-1}\\left(f_k^2 + \\frac{\\sigma_k^2}{s_k}\\right) - \\prod_{k=I-i}^{j-1}f_k^2 \\right)`
        Cuando :math:`i+j \\leq I`, devuelve 0.
    :Posibles errores:
        + **AssertionError**:
            + El parámetro :math:`i` no es un valor entero entre 0 e I.
            + El parámetro :math:`j` no es un valor entero entre 0 y J.
            + No se han calculado los suficientes valores de :math:`f_j`.
    """
    I, J = get_matrix_dimensions(triangle)
    assert i in range(I+1), "Período de incurrido a estimar inválido"
    assert j in range(J+1), "Período de reportado a estimar inválido"
    assert len(fjs_)+1 > j, "Se requieren más fj's calculados para estimar cij"
    if i + j <= I:
        return 0.0
    cij = triangle[i, I - i]
    fj_ = fjs_[I - i: j]
    sj_ = sjs_[I - i: j]
    sigma_sq = sigma_squares_[I - i: j]
    square_fjs = fj_ ** 2
    partial_res = prod(square_fjs + sigma_sq*1.0/sj_) - prod(square_fjs)
    return np.sqrt((cij**2)*partial_res)
def estimate_mack(triangle):
    """
        Estima :math:`C_{i,j}` para :math:`i+j > I` con el modelo teórico Mack y los errores asociados a
        la estimación. Finalmente genera un reporte estándar tal y como se detalla en :ref:`el output general <general_output>`.
        Esta función primero estima los parámetros requeridos para estimar la matriz con los :math:`C_{i, j}` y los
        errores asociados. Finalmente, llama a la función estimate_mack (respectivamente a las que estiman el error)
        para estimar :math:`C_{i,j}`.
    :param np.ndarray triangle: Matriz de información. Véase la sección :ref:`preparación de los datos de entrada <triangle_format>`
        para más información.
    :return: Salida estándar tal y como se describe en la sección :ref:`output general <general_output>`.
    :Posibles errores:
        + La estimación es **NaN**: Las entradas requeridas de la matriz para estimar :math:`f_j` son cero (o casi todas cero).
    """
    I, J = get_matrix_dimensions(triangle)
    scale_factor, triangle = scale_triangle(triangle)
    standard_triangle_check(triangle)
    fjs = np.array([fj(triangle, j) for j in range(J)])
    sjs = np.array([sj(triangle, j) for j in range(J)])
    cijs = np.array([[estimate_cij(triangle, fjs, i_, j_) for j_ in range(J + 1)] for i_ in range(I + 1)])
    sigma_squares = np.array([sigma_square(triangle, fjs, j) for j in range(J)])
    deviations_cijs = np.array([[cij_deviation(cijs, fjs, sigma_squares, i, j) for j in range(J + 1)] for i in range(I + 1)])
    deviations_xijs = np.array([[xijs_deviation(cijs, sigma_squares, fjs, deviations_cijs, i, j) for j in range(J+1) ] for i in range(I+1)])
    est_errors = np.array([[estimation_error(triangle, fjs, sjs, sigma_squares, i, j) for j in range(J + 1)] for i in range(I + 1)])
    prediction_errors = np.array([[prediction_error(deviations_cijs, est_errors, i, j) for j in range(J + 1)] for i in range(I + 1)])
    estimation_error_norm = vco_matrix(est_errors, cijs)
    process_error_norm = vco_matrix(prediction_errors, cijs)
    salida = gen_standard_output(scale_factor*cijs, fjs, scale_factor*deviations_cijs,
                                 scale_factor*deviations_xijs, scale_factor*est_errors)
    return salida