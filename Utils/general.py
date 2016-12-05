# coding: utf-8
import numpy as np
from warnings import warn
import json
"""
    Este archivo define varias utilidades comunes a todos los modelos.
"""
def xijs_from_cijs(cijs):
    """
        Calcula la matriz :math:`X` donde la entrada :math:`X_{i, j}` denota el pago por siniestros
        incurridos en el período :math:`i` reportados :math:`j` períodos después. Mas información en
        la sección :ref:`notación <notacion>`.
    :param np.ndarray cijs: Matriz acumulada de siniestros incurridos y reportados :math:`C_{i, j}`.
        Véase la sección :ref:`preparación de los datos de entrada <triangle_format>`
        para más información.
    :return: Matriz :math:`X` donde la :math:`i, j`-ésima entrada :math:`X_{i, j}` corresponde con
        :math:`X_{i, j} = C_{i, j} - C_{i, j-1}`.
    :Posibles errores:
        + **AssertionError**:
            + I es mayor que el número de columnas de la matriz triangle.
    """
    triangle_shift = np.zeros(cijs.shape)
    triangle_shift[:, 1:] = cijs[:, :-1]
    xijs = cijs - triangle_shift
    return xijs
def get_matrix_dimensions(triangle):
    """
        Para una matriz triangle de dimensión :math:`m \\times n`, los períodos :math:`I=m-1`,
        :math:`J=n-1` son los períodos hasta los cuáles implicitamente se tiene información.
    :param triangle: Matriz a la que se desea calcular los períodos implicitos.
    :return: I, J como se definieron arriba
    """
    return triangle.shape[0] - 1, triangle.shape[1] - 1
def prod(ls):
    """
        Multiplica los elementos de la lista ls.
    :param ls: Lista con valores numéricos
    :return: Multiplicación de todos los valores de la lista
    ls,
    """
    return reduce(lambda x, y: x * y, ls, 1)
def get_diagonal_from_matrix(matrix, n):
    """
        Dada una matriz :math:`A = [A_{i, j}]`, se define la :math:`n`-ésima diagonal de la matriz :math:`A`
        como la colección diag(A, n) = [:math:`C_{n, 0}, C_{n-1, 1}, \\cdots C_{n-k, k} \\cdots C_{n-J, J}` ]
        Esta función retorna la :math:`n`-ésima diagonal de la matriz matrix.
    :param matrix: Matriz a parir de la cuál extraer la diagonal.
    :param n: Diagonal a obtener.
    :return: :math:`n`-ésima diagonal de la matriz matrix.
    """
    I, J = get_matrix_dimensions(matrix)
    lista = []
    for i in range(min(I, n), max(n-J-1, -1), -1):
        j = n - i
        lista.append(matrix[i, j])
    return np.array(lista)
def msep(estimation_error, deviations):
    """
        Calcula el msep (min square error prediction) a partir del error de estimación
        y la varianza.
    :param estimation_error: Error de estimación.
    :param deviations: Desviaciones.
    :return: :math:`\\sqrt{\\text{Error de estimación}^2 + \\text{Desviaciones}^2}`
    """
    return estimation_error**2 + deviations**2
def prediction_error(deviations_, est_errors_, i, j):
    """
        Error de predicción tal y como se calcula en el apartado de :ref:`errores <errores>`.
        Cuando :math:`i+j \\leq I`, devuelve 0.
    :param deviations_: Matriz con las varianzas del modelo calculadas. Debe tener las mismas dimensiones
        que la :ref:`matriz de datos <triangle_format>`. La entrada :math:`i, j` debe ser la varianza
        calculada para la entrada :math:`i, j` de la :ref:`matriz de datos <triangle_format>`.
    :param est_errors_: Matriz con los errores de estimación calculadas. Debe tener las mismas dimensiones
        que la :ref:`matriz de datos <triangle_format>`. La entrada :math:`i, j` debe ser el error de estimación
        calculado para la entrada :math:`i, j` de la :ref:`matriz de datos <triangle_format>`.
    :param int i: Período en el que sucedió el siniestro. Toma valores desde 0 hasta :math:`I`.
    :param int j: Cantidad de períodos transcurridos después de sucedido el siniestro hasta que fue reportado.
        Toma valores desde 0 hasta :math:`J`.
    :return:  Cuando :math:`i+j > I`, devuelve:
        :math:`C_{i, I-i}^2\\left( \\prod_{j=I-i}^{I-1}\\left(f_j^2 + \\frac{\\sigma_j^2}{s_j}\\right) - \\prod_{j=I-i}^{I-1}f_j^2 \\right)`
        Cuando :math:`i+j \\leq I`, devuelve 0.
    :Posibles errores:
        + **AssertionError**:
            + El parámetro :math:`i` no es un valor entero entre 0 e I
            + El parámetro :math:`j` no es un valor entero entre 0 y J.
    """
    I, J = get_matrix_dimensions(deviations_)
    assert i in range(I+1), "Período de incurrido a estimar inválido"
    assert j in range(J+1), "Período de reportado a estimar inválido"
    return np.sqrt(deviations_[i, j] ** 2 + est_errors_[i, j] ** 2)
def standard_triangle_check(triangle):
    """
        Check básico para la :ref:`matriz de entrada <triangle_format>`.
        Comprueba si el triángulo esta compuesto de arrays de números crecientes, entradas no negativas y diagonal
        diferente de cero.
    :param triangle: :ref:`Matriz de entrada <triangle_format>`.
    :return:
    """
    if not all((triangle >= 0).ravel()):
        warn("Existen entradas negativas en el triángulo.")
    I, J = get_matrix_dimensions(triangle)
    diag = get_diagonal_from_matrix(triangle, I)
    if not all(diag > 1e-5):
        warn("Hay elementos de la diagonal que son cero.")
    for i in range(I):
        for j in range(1, min(I-i+1, J)):
            if not triangle[i, j-1] <= triangle[i, j]:
                warn("El triángulo no es acumulado.")
def scale_triangle(triangle):
    """
        Escala la matriz triángulo para que quede con sólo una cifra entera.
        Retorna el factor de escala (valor por el que se tiene que multiplicar
        para tener la matriz inicial) y la matriz escalada.
    :param np.ndarray triangle: Matriz a escalar.
    :return:
    """
    triangle = np.array(triangle, dtype=float)
    cents = int(np.log10(triangle.max()))
    scale_factor = 10.0**cents
    triangle /= (1.0*scale_factor)
    return scale_factor, triangle
def vco_vector(deviations_vector, data_vector):
    """
        El coeficiente de variacion o vco se define como :math:`\\frac{\\text{\\sigma}}{\\mu}` donde
        :math:`\\sigma = \\text{desviación}` y :math:`\\mu = \\text{estimación}`. Esta función recibe
        un vector de desviaciones :math:`\\sigma_1, \\cdots, \\sigma_n` y un vector de estimaciones
        :math:`\\mu_1, \\cdots, \\mu_n` y devuelve el vector :math:`\\frac{\\sigma_i}{\\mu_i}`.
    :param deviations_vector: Array de desviaciones :math:`\\sigma_1, \\cdots, \\sigma_n`.
    :param data_vector: Array de estimación :math:`\\mu_1, \\cdots, \\mu_n`.
    :return: Array :math:`\\frac{\\sigma_i}{\\mu_i}`.
    """
    J = len(deviations_vector)-1
    return np.array([deviations_vector[j] * 1.0 / data_vector[j] for j in range(J+1)])
def vco_matrix(deviations_matrix, data_matrix):
    """
        El coeficiente de variacion o vco se define como :math:`\\frac{\\text{\\sigma}}{\\mu}` donde
        :math:`\\sigma = \\text{desviación}` y :math:`\\mu = \\text{estimación}`. Esta función recibe
        una matriz de desviaciones :math:`\\sigma_{i, j}` y una matriz de estimaciones
        :math:`\\mu_{i, j}` y devuelve la matriz :math:`\\frac{\\sigma_{i, j}}{\\mu_{i, j}}`.
    :param deviations_matrix: Matriz de desviaciones :math:`\\sigma_{i, j}`.
    :param data_matrix: Matriz de estimación :math:`\\mu_{i, j}`.
    :return: Matriz :math:`\\frac{\\sigma_{i, j}}{\\mu_{i, j}}`.
    """
    I, J = get_matrix_dimensions(deviations_matrix)
    return np.array([[deviations_matrix[i, j] * 1.0 / data_matrix[i, j] for j in range(J+1)] for i in range(I+1)])
def vco_reserves(cijs, deviations_cijs):
    """
    El coeficiente de variacion o vco se define como :math:`\\frac{\\text{\\sigma}}{\\mu}` donde
    :math:`\\sigma = \\text{desviación}` y :math:`\\mu = \\text{estimación}`. Esta función calcula
    el vco de las reservas :math:`R_j` (:math:`R_j` se define en la sección  :ref:`notación <notation>`)
    a partir de los :math:`C_{i, j}` y sus desviaciones.
    :param cijs: Matriz acumulada de reservas :math:`C_{i, j}`.
    :param deviations_cijs: Matriz de desviaciones :math:`\\sigma_{i, j}` de :math:`C_{i, j}`.
    :return: Array de numpy donde la entrada :math:`i` corresponde con 0 para :math:`i=0`, y para :math:`i=1,\\cdots,I`
        corresponde con
        :math:`\\frac{\\sigma_{i, J}}{C_{i, J} - C_{i, I-i}`
        donde :math:`J` es el último período de información sobre reportados.
    """
    I, J = get_matrix_dimensions(cijs)
    return np.array([0.0] + [deviations_cijs[i, J] * 1.0 / (cijs[i, -1] - cijs[i, I - i]) for i in range(I-J+1, I+1)])
def fj(triangle, j):
    """
        Estimación del factor de desarrollo :math:`f_j`.
    :param np.ndarray triangle: Matriz de información. Véase la sección :ref:`preparación de los datos de entrada <triangle_format>`
        para más información.
    :param int j: Entero con valor entre 0 y :math:`J-1` (incluyendo) donde :math:`J` es el número de columnas
        de la matriz triangle.
    :return: Para :math:`j=0,1,\cdots, J-1`, donde :math:`J` el número de columnas de la matriz triangle, devuelve el cálculo de
        :math:`f_{j} = \\frac{\\sum_{k=0}^{I-j}C_{k, j}}{\sum_{k=0}^{I - j}C_{k, j-1}}`
    :Posibles errores:
        + **AssertionError**:
            + El parámetro :math:`j` no es un valor entero entre 0 y J-1
        + La estimación es **NaN**: Las entradas requeridas de la matriz para estimar :math:`f_j` son cero (o casi todas cero).
    """
    I, J = get_matrix_dimensions(triangle)
    assert j in range(J), u"j debe ser un valor entero entre 0 y J-1, donde J es el número de columnas" \
                          u" de la matriz."
    cij = triangle[:I - j, j]
    cij_next = triangle[:I - j, j + 1]
    assert abs(sum(cij)) > 0.00000001, u"No es posible calcular el factor de desarrollo {} para este caso.".format(j)
    return sum(cij_next) * 1.0 / sum(cij)
def fj_deviation(cijs, sigma_squares, j):
    """
        Estima la desviación del parámetro :math:`f_j`.
    :param np.ndarray cijs: Matriz del mismo tamaño que el :ref:`triángulo inicial <triangle_format>`
        pero con los valores restantes ya estimados, y donde cada entrada corresponde con :math:`C_{i, j}`.
        Como :math:`C_{i, j}` para :math:`i+j \leq I` son valores ya dados, estas entradas deben
        ser iguales en ambas matrices (cijs y el :ref:`triángulo inicial <triangle_format>` ).
    :param np.ndarray sigma_squares: Lista de tamaño :math:`J` de parámetros :math:`\\sigma_j`
        para :math:`j = 0, 1\\cdots J-1` ya estimados.
    :param int j: Valor que indica que será estimada la :math:`j`-ésima desviación del parámetro :math:`f_j`. Puede
        tomar los valores :math:`j = 0, 1\\cdots J-1`.
    :return: Cálculo de
        :math:`\\text{desviación de }f_j = \\sqrt{\\frac{\\sigma_j^2}{\\sum_{i=0}^{I-j-1}C_{i, j}}}`
    """
    I, J = get_matrix_dimensions(cijs)
    assert j in range(J), "j debe ser un valor entero entre 0 y J-1, donde J es el número de columnas" \
                          " de la matriz triángulo menos 1."
    return np.sqrt(sigma_squares[j]*1.0/sum(cijs[:I-j, j]))
def replace_nan(element):
    """
        Reemplaza un elemento si es NaN (np.NaN) por "NaN"*[]:
    :param element: Elemento a verificar si es NaN para reemplazarlo.
    :return: Si element es NaN, retorna "NaN". Si no, retorna element.
    """
    return None if not element < np.inf else element
def sigma_square(triangle, fjs_, j):
    """
        Estimador del parámetro :math:`\sigma_j^2` requerido para calcular la varianza del proceso y el
        error de estimación.
    :param np.ndarray fjs_: Estimación de :math:`f_j`. La lista fjs\_ debe
        contener las entradas computadas de los parámetros :math:`f_j` desde 0 hasta j-1, donde
        se asume que la :math:`j`-ésima entrada de fjs\_ corresponde con :math:`f_j` (:math:`\\text{fjs_[j]} = f_j`).
    :param np.ndarray triangle: Matriz de información. Véase la sección :ref:`preparación de los datos de entrada <triangle_format>`
        para más información.
    :param int j: Entero con valor entre 0 y :math:`J-1` (incluyendo) donde :math:`J` es el número de columnas
        de la matriz triangle.
    :return:
        Si :math:`I - j - 1 > 0`, se aplica la fórmula:
        :math:`\\qquad\\sigma_j^2 = \\frac{1}{I-(j+1)}\\sum_{i=0}^{I-(j+1)}C_{i, j}\\left( \\frac{C_{i, j+1}}{C_{i, j}} - f_j\\right)^2`
        si :math:`I - j - 1 = 0`, computamos:
        :math:`\\qquad\\sigma_j^2 = \\min\\left\\{\\frac{\\sigma_{I-2}^4}{\\sigma_{I-3}^2}, \\sigma_{I-3}^2, \\sigma_{I-2}^2\\right\\}`
        donde :math:`\\sigma_{I-2}` y :math:`\\sigma_{I-3}` se calculan con esta misma función (sigma_square)
        con :math:`j=I-2` y :math:`j=I-3` respectivamente. Para :math:`I - j - 1 < 0`, :math:`\\sigma_j` no está definido.
    :Posibles errores:
        + **AssertionError**:
            + El parámetro :math:`j` no es un valor entero entre 0 y J.
    """
    I, J = get_matrix_dimensions(triangle)
    if I <= 2:
        return np.NaN
    assert j in range(J), "j debe ser un valor entero entre 0 y J-1, donde J es el número de columnas" \
                          " de la matriz triángulo menos 1."
    assert len(fjs_)+1 > j, "Se requieren más fj's calculados para estimar cij"
    if (I - j - 1) > 0:
        cij = triangle[:I - j, j]
        cijp1 = triangle[:I - j, j + 1]
        fj_ = fjs_[j]
        index = [i for i in range(len(cij)) if cij[i] != 0]
        p1 = (cij * (cijp1 * 1.0 / cij - fj_) ** 2.0)[index]
        return 1.0/(I - j - 1) * (sum(p1))
    elif (I - j - 1) == 0:
        s1 = sigma_square(triangle, fjs_, j - 1)
        s2 = sigma_square(triangle, fjs_, j - 2)
        if abs(s1 < 0.0001) or abs(s2 < 0.0001):
            return 0.0
        return min((s1 ** 2) * 1.0 / s2, s1, s2)
def gen_standard_output(cijs, fjs, deviations_cijs, deviations_xijs=None, estimation_error_cijs_abs=None, info=None,
                        include_development_factor_errors=True):
    """
        Esta función esta diseñada para generar un output estándar para cada uno de los modelos. Lo mínimo requerido
        es la matriz de :math:`C_{i, j}` (cijs), la matriz de :math:`X_{i, j}` (xijs), la lista de factores de desarrollo
        :math:`f_j` (fjs) y la matriz de desviaciones de :math:`C_{i, j}` (deviations_cijs).
    :param np.ndarray cijs: Matriz del mismo tamaño que el :ref:`triángulo inicial <triangle_format>`
        pero con los valores restantes ya estimados, y donde cada entrada corresponde con :math:`C_{i, j}`.
    :param fjs: Estimación de los factores de desarrollo :math:`f_j`. La lista fjs debe
        contener las entradas computadas de los parámetros :math:`f_j` desde 0 hasta :math:`j-1`, donde
        se asume que la :math:`j`-ésima entrada de fjs corresponde con :math:`f_j` (:math:`\\text{fjs[j]} = f_j`).
    :param deviations_cijs: Matriz del mismo tamaño que la matriz cijs donde la entrada :math:`i, j` corresponde
        con la desviación del parámetro :math:`C_{i, j}`.
    :param deviations_xijs: Matriz del mismo tamaño que la matriz xijs donde la entrada :math:`i, j` corresponde
        con la desviación del parámetro :math:`X_{i, j}`.
    :param info: Información adicional.
    :return: Salida estándar tal y como se describe en la sección :ref:`output general <general_output>`.
    """
    I, J = get_matrix_dimensions(cijs)
    sigma_squares = np.array([sigma_square(cijs, fjs, j) for j in range(J)])
    if include_development_factor_errors:
        deviations_fjs = np.array([fj_deviation(cijs, sigma_squares, j) for j in range(J)])
    else:
        deviations_fjs = np.empty(J)
        deviations_fjs[:] = np.NaN
    assert I >= J, "La cantidad de columnas del triángulo debe ser a lo sumo la cantidad de filas del mismo."
    assert deviations_cijs.shape[0] == cijs.shape[0], "No coinciden los tamaños de la matriz de desviación y " \
                                                      "la matriz de estimación"
    assert len(fjs) == J
    xijs = xijs_from_cijs(cijs)
    reserves = [cijs[i, -1] - cijs[i, I - i] for i in range(I-J, I + 1)]
    cv_reserves = vco_reserves(cijs, deviations_cijs)
    coefficient_of_variation_cij = vco_matrix(deviations_cijs, cijs)
    coefficient_of_variation_fj = vco_vector(deviations_fjs, fjs)
    if info is None:
        info = list()
    if deviations_xijs is not None:
        assert deviations_xijs.shape == cijs.shape
        coefficient_of_variation_xij = vco_matrix(deviations_xijs, xijs)
    else:
        deviations_xijs = np.empty((I+1, J+1,))
        deviations_xijs[:] = "NaN"
        coefficient_of_variation_xij = np.empty((I+1, J+1,))
        coefficient_of_variation_xij[:] = "NaN"
    if estimation_error_cijs_abs is not None:
        assert estimation_error_cijs_abs.shape[0] == cijs.shape[0]
        estimation_error_cijs_rel = vco_matrix(estimation_error_cijs_abs, cijs)
    else:
        estimation_error_cijs_rel = np.empty((I + 1, J + 1))
        estimation_error_cijs_rel[:] = "NaN"
        estimation_error_cijs_abs = np.empty((I + 1, J + 1))
        estimation_error_cijs_abs[:] = "NaN"
    msep_cijs = msep(estimation_error_cijs_rel, deviations_cijs)
    cv_msep_cijs = vco_matrix(np.sqrt(msep_cijs), cijs)
    development_periods_cij = np.array([sum(get_diagonal_from_matrix(cijs, n)) for n in range(I, I + J + 1)])
    development_periods_deviations_cij = np.sqrt(np.array([sum(get_diagonal_from_matrix(deviations_cijs, n) ** 2) for n in range(I, I + J + 1)]))
    cv_development_periods_cij = vco_vector(development_periods_deviations_cij, development_periods_cij)
    development_periods_xij = np.array([sum(get_diagonal_from_matrix(xijs, n)) for n in range(I, I + J + 1)])
    development_periods_deviations_xij = np.sqrt(np.array([sum(get_diagonal_from_matrix(deviations_xijs, n) ** 2) for n in range(I, I + J + 1)]))
    cv_development_periods_xij = vco_vector(development_periods_deviations_xij, development_periods_xij)
    occurrence_periods_msep = msep(estimation_error_cijs_rel[I - J:, -1], deviations_cijs[I - J:, -1])
    occurrence_periods_msep_norm = vco_vector(np.sqrt(occurrence_periods_msep), cijs[I-J:, -1])
    matriz = [[{'cij': {'est': replace_nan(cijs[i, j]),
                        'vco': replace_nan(coefficient_of_variation_cij[i, j]),
                        'vco_abs': replace_nan(deviations_cijs[i, j]),
                        'est_err': replace_nan(estimation_error_cijs_rel[i, j]),
                        'est_err_abs': replace_nan(estimation_error_cijs_abs[i, j]),
                        'msep': replace_nan(cv_msep_cijs[i, j]),
                        'msep_abs': replace_nan(msep_cijs[i, j])},
                'xij': {'est': replace_nan(xijs[i, j]),
                        'vco': replace_nan(coefficient_of_variation_xij[i, j]),
                        'vco_abs': replace_nan(deviations_xijs[i, j])}}
               for j in range(J + 1)]
              for i in range(I + 1)]
    by_calendar_period = [{'xij': {'est': replace_nan(development_periods_xij[j]),
                                   'vco': replace_nan(cv_development_periods_xij[j]),
                                   'vco_abs': replace_nan(development_periods_deviations_xij[j])}}
                          for j in range(J + 1)]
    by_occurrence_period = [{'cij': {'est': replace_nan(cijs[i, -1]),
                                     'vco': replace_nan(coefficient_of_variation_cij[i, -1]),
                                     'vco_abs': replace_nan(deviations_cijs[i, -1]),
                                     'est_err': replace_nan(estimation_error_cijs_rel[i, -1]),
                                     'est_err_abs': replace_nan(estimation_error_cijs_abs[i, -1]),
                                     'msep': replace_nan(occurrence_periods_msep_norm[i-I+J]),
                                     'msep_abs': replace_nan(occurrence_periods_msep[i-I+J])},
                             'ri': {'est': replace_nan(reserves[i-I+J]),
                                    'vco': replace_nan(cv_reserves[i-I+J]),
                                    'vco_abs': replace_nan(deviations_cijs[i-I+J, -1]),
                                    'est_err': replace_nan(estimation_error_cijs_rel[i, -1]),
                                    'est_err_abs': replace_nan(estimation_error_cijs_abs[i, -1]),
                                    'msep': replace_nan(occurrence_periods_msep_norm[i-I+J]),
                                    'msep_abs': replace_nan(occurrence_periods_msep[i-I+J])}}
                            for i in range(I-J, I+1)]
    development_factors = [{'fj': {'est': replace_nan(fjs[j]),
                                   'vco': replace_nan(coefficient_of_variation_fj[j]),
                                   'vco_abs': replace_nan(deviations_fjs[j])}} for j in range(J)]
    vco_projection_rel = np.sqrt(sum(deviations_cijs[:, -1]**2))*1.0/sum(cijs[:, -1])
    vco_projection_abs = np.sqrt(sum(deviations_cijs[:, -1]**2))
    total_projection_cij = {'est': replace_nan(sum([cijs[i, -1] for i in range(I-J, I+1)])),
                            'vco': replace_nan(vco_projection_rel),
                            'vco_abs': replace_nan(vco_projection_abs)}
    total_projection_ri = {'est': replace_nan(sum([reserves[i-I+J] for i in range(I-J, I+1)])),
                           'vco': replace_nan(vco_projection_rel),
                           'vco_abs': replace_nan(vco_projection_abs)}
    salida = {"matriz": matriz,
              "by_calendar_period": by_calendar_period,
              "by_occurrence_period": by_occurrence_period,
              "development_factors": development_factors,
              "total_projection_cij": total_projection_cij,
              "total_projection_ri": total_projection_ri,
              "info": info}

    salida = {"matriz": matriz,
              "byCalendarPeriod": by_calendar_period,
              "byOccurrencePeriod": by_occurrence_period,
              "developmentFactors": development_factors,
              "totalProjectionCij": total_projection_cij,
              "totalProjectionRi": total_projection_ri,
              "info": info}
    return salida