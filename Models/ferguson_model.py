# coding: utf-8
import numpy as np
from Utils import prod, vco_matrix, fj, get_matrix_dimensions, \
    gen_standard_output, xijs_from_cijs, standard_triangle_check, \
    prediction_error, scale_triangle
from mack_model import sigma_square
def estimate_cij(triangle, fjs_, i, j, nus):
    """
        Estimación de :math:`C_{i, j}` mediante el modelo Ferguson interpretado como el pago acumulado de los siniestros
        incurridos en el período :math:`i` reportados hasta :math:`j` períodos después.
        El modelo Ferguson requiere de una estimación *a priori* por parte de un experto
        de :math:`C_{i, J}` (:math:`J` es el último período de desarrollo).
        En otras palabras se requiere de una estimación de los pagos acumulados de siniestros
        incurridos en cada período y reportados el último período de registro, en este caso, :math:`J`.
        Esta estimación se denota por :math:`\\nu_i = C_{i, J}`.
    :param np.array fjs_: Estimación de los factores de desarrollo :math:`f_j`. La lista fjs\_ debe
        contener las entradas computadas de los parámetros :math:`f_j` desde 0 hasta :math:`j-1`, donde
        se asume que la :math:`j`-ésima entrada de fjs\_ corresponde con :math:`f_j` (:math:`\\text{fjs_[j]} = f_j`).
    :param np.ndarray triangle: Matriz de información. Véase la sección :ref:`preparación de los datos de entrada <triangle_format>`
        para más información.
    :param int i: Período en el que sucedió el siniestro. Toma valores desde 0 hasta :math:`I`.
    :param int j: Cantidad de períodos transcurridos después de sucedido el siniestro hasta que fue reportado.
        Toma valores desde 0 hasta :math:`J`.
    :param np.array nus:
        Estimación *a priori* del parámetro :math:`\\nu_i`. Es una lista de tamaño :math:`I`, donde la :math:`i`-ésima entrada
        es la estimación *a priori* por parte de un experto de :math:`C_{i, J}` (pago de siniestros incurridos en el período :math:`i`
        reportados en el período :math:`J`).
    :return:
        Cuando :math:`i+j>I`, se calcula mediante la fórmula
        :math:`C_{i, j} = \\beta_jC_{i, J}`
        donde
        :math:`C_{i, J} = C_{i, I-i} + (1- \\beta_{I-i})\\nu_i`
        y donde :math:`\\beta_j=\\prod_{k=j}^{J-1}\\frac{1}{f_k}`.
        Cuando :math:`i+j \\leq I`, devuelve la posición :math:`i, j` de la matriz triangle.
    :Posibles errores:
        + **AssertionError**:
            + El parámetro :math:`i` no es un valor entero entre 0 e I.
            + El parámetro :math:`j` no es un valor entero entre 0 y J.
            + No se han calculado los suficientes valores de :math:`f_j`.
            + La lista :math:`\\mu_i` debe tener longitud al menos de i+1.
    """
    I, J = get_matrix_dimensions(triangle)
    assert i in range(I+1), "Período de incurrido a estimar inválido"
    assert j in range(J+1), "Período de reportado a estimar inválido"
    assert len(fjs_)+1 > j, "Se requieren más fj's calculados para estimar cij"
    assert len(nus) > i, "se requieren mas estimacion es de ciJ para calcular cij"
    if i+j <= I:
        return triangle[i, j]
    beta_I_i = prod(1.0 / fjs_[I - i: J])
    ciJ = triangle[i, I - i] + (1 - beta_I_i) * nus[i]
    betaj = prod(1.0 / fjs_[j: J])
    return betaj * ciJ
def gamma(fjs_, j, J):
    """
        Estimación del parámetro :math:`\\gamma_j` requerido para calcular el error de estimación.
    :param np.array fjs_: Estimación de los factores de desarrollo :math:`f_j`. La lista fjs\_ debe
        contener las entradas computadas de los parámetros :math:`f_j` desde 0 hasta :math:`J-1`, donde
        se asume que la :math:`j`-ésima entrada de fjs\_ corresponde con :math:`f_j` (:math:`\\text{fjs_[j]} = f_j`).
    :param int j: Entero con valor entre 0 y :math:`J` (incluyendo 0 y :math:`J`).
    :param int J: Número de columnas de la :ref:`matriz de datos <triangle_format>`.
    :return: Si :math:`j = 0`, retorna :math:`\\gamma_0 = \\prod_{k=0}^{J-1}f_j`.
        Si :math:`j = J`, retorna :math:`\\gamma_J = 1-\\sum_{n=0}^{J-1}\\gamma_n` donde los parámetros
        :math:`\\gamma_n` se estiman con esta misma función.
        Si :math:`j = 1, \\cdots, J-1`, retorna :math:`\\gamma_j = \\left(1-\\frac{1}{f_{j-1}}\\right)\\prod_{k=j}^{J-1}\\frac{1}{f_k}`.
    :Posibles errores:
        + **AssertionError**:
            + El parámetro :math:`j` no es un valor entero entre 0 y  :math:`J`
            + La longitud de la lista fjs\_ es menor que :math:`J`.
    """
    assert j in range(J+1), "j debe ser menor o igual a J."
    assert len(fjs_) >= J, "No hay suficientes fjs para estimar gamma_j."
    if j == 0:
        beta = prod(1.0 / fjs_[: J])
        return beta
    elif j == J:
        return 1 - sum([gamma(fjs_, a, J) for a in range(J)])
    else:
        return prod(1.0 / fjs_[j: J]) * (1.0 - 1.0 / fjs_[j - 1])
def calc_mu(triangle, fjs_, i):
    """
        Estimador del parámetro :math:`\\mu_i` requerido para la calcular el error de estimación.
    :param np.ndarray triangle: Matriz de información. Véase la sección :ref:`preparación de los datos de entrada <triangle_format>`
        para más información.
    :param np.array fjs_: Estimación de los factores de desarrollo :math:`f_j`. La lista fjs\_ debe
        contener las entradas computadas de los parámetros :math:`f_j` desde 0 hasta :math:`I-1`, donde
        se asume que la :math:`j`-ésima entrada de fjs\_ corresponde con :math:`f_j` (:math:`\\text{fjs_[j]} = f_j`).
    :param int i: Entero con valor entre 0 e :math:`I-1` (incluyendo 0 e :math:`I-1`).
    :return: :math:`\\mu_i = C_{i, I-i}\\prod_{k=I-i}^{I-1}f_k`.
    :Posibles errores:
        + **AssertionError**:
            + El parámetro :math:`i` no es un valor entero entre 0 e I.
            + No se han calculado los suficientes valores de :math:`f_j`.
    """
    I, J = get_matrix_dimensions(triangle)
    assert i in range(I+1), "Período de incurrido a estimar inválido"
    assert len(fjs_) >= I, "Se requieren más fj's calculados para estimar cij"
    return triangle[i, I - i] * prod(fjs_[I - i: I])
def mij(gammas_, mus, i, j):
    """
        Estimador del parámetro :math:`m_{i, j}` requerido para la calcular el error de estimación.
    :param np.array gammas_: Estimación de parámetros :math:`\\gamma_j`. La lista gammas\_ debe
        contener las entradas computadas de los parámetros :math:`\\gamma_j` desde 0 hasta :math:`j+1`, donde
        se asume que la :math:`j`-ésima entrada de gammas\_ corresponde con :math:`\\gamma_j` (:math:`\\text{gammas_[j]} = \\gamma_j`).
    :param mus: Estimación de los parámetros :math:`\\mu_i`. La lista nus debe
        contener las entradas computadas de los parámetros :math:`\\mu_i` desde 0 hasta :math:`i+1`, donde
        se asume que la :math:`i`-ésima entrada de mus corresponde con :math:`\\mu_i` (:math:`\\text{mus[j]} = \\mu_j`).
    :param int i: Entero con valor entre 0 e :math:`I` (incluyendo 0 e :math:`I`) donde :math:`I` es el período hasta el que
        se tiene información.
    :param int j: Entero con valor entre 0 e :math:`I` (incluyendo 0 e :math:`I`) donde :math:`I` es el período hasta el que
        se tiene información.
    :return: :math:`m_{i, j} = \\mu_i\\gamma_j`.
    :Posibles errores:
        + **AssertionError**:
            + La longitud de la lista gammas\_ debe ser al menos j+1.
            + La longitud de la lista nus debe ser al menos i+1.
    """
    assert len(gammas_) >= j, "Se deben computar mínimo j gammas para poder estimar este parámetros"
    assert len(mus) >= i, "Se deben computar mínimo i nus para poder estimar este parámetros"
    return gammas_[j] * mus[i]
def d(I):
    """
        Cálculo de :math:`d`, parámetro requerido para estimar :math:`\\phi`.
    :param int I: Período hasta el cual se dispone de información.
    :return: :math:`\\frac{(I+1)(I+2)}{2} - 2I-1`.
    """
    return (I + 1) * (I + 2) * 1.0 / 2 - 2 * I - 1
def single_payments_from_triangle(triangle):
    """
        Calcula la matriz :math:`X` donde la entrada :math:`X_{i, j}` denota el pago por siniestros
        incurridos en el período :math:`i` reportados :math:`j` períodos después. Mas información sobre la definición
        de :math:`X_{i, j}` en la sección :ref:`notación <notacion>`. Esta función esta pensada para
        calcular la matriz de pagos simples a partir de la matriz triángulo de datos, por lo que las entradas
        desconocidas :math:`i+j > I` continuan siendo desconocidas y son cero.
    :param np.ndarray triangle: Matriz de información. Véase la sección :ref:`preparación de los datos de entrada <triangle_format>`
        para más información.
    :return: Matriz :math:`X` donde la :math:`i, j`-ésima entrada :math:`X_{i, j}` corresponde con
        Si :math:`i+j\\leq I`, retorna
        :math:`X_{i, j} = C_{i, j} - C_{i, j-1}`.
        Si :math:`i+j > I`, retorna 0.
    :Posibles errores:
        + **AssertionError**:
    """
    I, J = get_matrix_dimensions(triangle)
    triangle_shift = np.zeros(triangle.shape)
    triangle_shift[:, 1:] = triangle[:, :-1]
    xijs = triangle - triangle_shift
    for i in range(I+1):
        for j in range(J+1):
            if i + j > I:
                xijs[i, j] = 0
    return xijs
def phi(xijs_, d, mijs_):
    """
        Estimador del parámetro :math:`\\phi`.
    :param xijs_: Matriz :math:`X` donde la entrada :math:`X_{i, j}` denota el pago por siniestros
        incurridos en el período :math`i` reportados :math:`j` períodos después. Mas información en
        la sección :ref:`notación <notacion>`.
    :param d: Cómputo de :math:`d`.
    :param mijs_: Matriz de tamaño :math:`I\\times I` donde la :math:`i, j`-ésima entrada corresponde con el parámetro
        :math:`m_{i, j}` (:math:`\\text{mij_[i, j]} = m_{i, j}`).
    :return:  :math:`\\phi = \\frac{1}{d}\\sum_{i+j\\leq I}\\frac{(X_{i, j} - m_{i, j})^2}{m_{i, j}}`
    :Posibles errores:
        + **AssertionError**:
    """
    I, J = get_matrix_dimensions(xijs_)
    assert mijs_.shape == xijs_.shape, "Las matrices xijs y mijs deben tener el mismo tamaño"
    comb = [(i, j) for i in range(I + 1) for j in range(J + 1) if i + j <= I and abs(mijs_[(i, j)]) > 0.00001]
    xij = np.array([xijs_[a] for a in comb])
    mij = np.array([mijs_[a] for a in comb])
    return (1.0/d)*sum(((xij - mij) ** 2)*1.0/mij)
def h1(phi_, gammas_, nus, i, I):
    """
        Cálculo de :math:`h_{i+1, i+1}` requerido para estimar la
        matriz de información de Fisher.
    :param phi_: Parámetro :math:`\\phi`.
    :param np.ndarray gammas_: Estimación de parámetros :math:`\\gamma_j`. La lista gammas\_ debe
        contener las entradas computadas de los parámetros :math:`\\gamma_j` desde 0 hasta :math:`I-i`, donde
        se asume que la :math:`j`-ésima entrada de gammas\_ corresponde con :math:`\\gamma_j` (:math:`\\text{gammas_[j]} = \\gamma_j`).
    :param nus: Estimación de parámetros :math:`\\nu_i`. La lista nus debe
        contener las entradas computadas de los parámetros :math:`\\nu_i` desde 0 hasta :math:`i+1`, donde
        se asume que la :math:`i`-ésima entrada de nus corresponde con :math:`\\nu_i` (:math:`\\text{nus[j]} = \\nu_j`).
    :param int i: Entero con valor entre 0 e :math:`I` (incluyendo 0 e :math:`I`) donde :math:`I` es el período hasta el que
        se tiene información.
    :param int I: Período hasta el cual se dispone de información.
    :return: :math:`h_{i+1, i+1} = \\frac{1}{\\nu_i \\phi}\\sum_{j=0}^{I-i}\\gamma_j`.
    """
    return 1.0/(nus[i] * phi_) * sum(gammas_[: I - i + 1])
def h2(phi_, gammas_, nus, j, I):
    """
        Cálculo de :math:`h_{I+2+j, I+2+j}` requerido para estimar la
        matriz de información de Fisher.
    :param phi_: Parámetro :math:`\\phi`.
    :param np.ndarray gammas_: Estimación de parámetros :math:`\\gamma_j`. La lista gammas\_ debe
        contener las entradas computadas de los parámetros :math:`\\gamma_j` desde 0 hasta :math:`I`, donde
        se asume que la :math:`j`-ésima entrada de gammas\_ corresponde con :math:`\\gamma_j` (:math:`\\text{gammas_[j]} = \\gamma_j`).
    :param nus: Estimación de parámetros :math:`\\nu_i`. La lista nus debe
        contener las entradas computadas de los parámetros :math:`\\nu_i` desde 0 hasta :math:`I-j`, donde
        se asume que la :math:`i`-ésima entrada de nus corresponde con :math:`\\nu_i` (:math:`\\text{nus[j]} = \\nu_j`).
    :param int j: Entero con valor entre 0 e :math:`I-1` (incluyendo 0 e :math:`I-1`) donde :math:`I` es el período hasta el que
        se tiene información.
    :param int I: Período hasta el cual se dispone de información.
    :return: :math:`h_{I+2+j, I+2+j} = \\frac{1}{\\gamma_j \\phi}\\sum_{i=0}^{I-j}\\mu_i + \\frac{\\nu_0}{\\phi(1-\\sum_{n=0}^{I-1}\\gamma_n)}`.
    """
    s1 = 1.0/(gammas_[j] * phi_)*sum(nus[:I - j + 1])
    s2 = nus[0] * 1.0 / (phi_ * (1.0 - sum(gammas_[:I])))
    return s1 + s2
def h3(phi_, gammas_, nus, j, l, I):
    """
        Cálculo de :math:`h_{I+2+j, I+2+l}` requerido para estimar la
        matriz de información de Fisher.
    :param phi_: Parámetro :math:`\\phi`.
    :param gammas_: Estimación de parámetros :math:`\\gamma_j`. La lista gammas\_ debe
        contener las entradas computadas de los parámetros :math:`\\gamma_j` desde 0 hasta :math:`I`, donde
        se asume que la :math:`j`-ésima entrada de gammas\_ corresponde con :math:`\\gamma_j` (:math:`\\text{gammas_[j]} = \\gamma_j`).
    :param nus: Estimación de parámetros :math:`\\nu_i`. La lista nus debe
        contener las entradas computadas de los parámetros :math:`\\nu_i` desde 0 hasta :math:`I-j`, donde
        se asume que la :math:`i`-ésima entrada de nus corresponde con :math:`\\nu_i` (:math:`\\text{nus[j]} = \\nu_j`).
    :param int j: Entero con valor entre 0 e :math:`I-1` (incluyendo 0 e :math:`I-1`) donde :math:`I` es el período hasta el que
        se tiene información con :math:`j \\neq l`.
    :param int l: Entero con valor entre 0 e :math:`I-1` (incluyendo 0 e :math:`I-1`) donde :math:`I` es el período hasta el que
        se tiene información con :math:`j \\neq l`.
    :param int I: Período hasta el cual se dispone de información.
    :return: :math:`h_{I+2+j, I+2+l} = \\frac{\\nu_0}{\\phi(1-\\sum_{n=0}^{I-1}\\gamma_n)}`.
    """
    assert j != l, 'Invalid formula to calculate the matrix'
    return nus[0] * 1.0 / (phi_ * (1 - sum(gammas_[:I])))
def fisher_matrix(phi_, h1_, h2_, h3_, I):
    """
        Estimador de la matriz de información de Fisher para el cálculo del error de
        estimación.
    :param phi_: Parámetro :math:`\\phi`.
    :param np.array h1_: Estimación de :math:`h_{i+1, i+1}` estimados para :math:`i` desde :math:`0` hasta :math:`I`. La lista h1\_ debe
        contener las entradas computadas de los parámetros :math:`h_{i+1, i+1}` desde 0 hasta :math:`I`, donde
        se asume que la :math:`i`-ésima entrada de h1\_ corresponde con :math:`h_{i+1, i+1}` (:math:`\\text{h1_[i]} = h_{i+1, i+1}`).
    :param np.array h2_: Estimación de :math:`h_{I+2+j, I+2+j}` estimados para :math:`j` desde :math:`0` hasta :math:`I`. La lista h2\_ debe
        contener las entradas computadas de los parámetros :math:`h_{I+2+j, I+2+j}` desde 0 hasta :math:`I`, donde
        se asume que la :math:`i`-ésima entrada de h2\_ corresponde con :math:`h_{I+2+j, I+2+j}` (:math:`\\text{h2_[i]} = h_{I+2+j, I+2+j}`).
    :param np.ndarray h3_: Matriz de :math:`h_{I+2+j, I+2+l}` estimados para :math:`j, l` desde :math:`0` hasta :math:`I`. La lista h3\_ debe
        contener las entradas computadas de los parámetros :math:`h_{I+2+j, I+2+l}` desde 0 hasta :math:`I`, donde
        se asume que la :math:`(j, l)`-ésima entrada de h3\_ corresponde con :math:`h_{I+2+j, I+2+l}` (:math:`\\text{h3_[i]} = h_{I+2+j, I+2+l}`).
    :param int I: Período hasta el cual se dispone de información.
    :return: Se denota por :math:`\\mathscr{H}` la matriz de información de fisher de tamaño :math:`(2I+1) \\times (2I+1)` y
        :math:`h_{i, j}` las entradas de esta matriz.
        Entonces,
        :math:`h_{i+1, i+1} = \\frac{1}{\\nu_i \\phi}\\sum_{j=0}^{I-i}\\gamma_j` para :math:`i=0, 1,\\cdots, I`.
        :math:`h_{I+2+j, I+2+j} = \\frac{1}{\\gamma_j \\phi}\\sum_{i=0}^{I-j}\\mu_i + \\frac{\\nu_0}{\\phi(1-\\sum_{n=0}^{I-1}\\gamma_n)}` para :math:`j=0, 1,\\cdots, I-1`.
        :math:`h_{I+2+j, I+2+l} = \\frac{\\nu_0}{\\phi(1-\\sum_{n=0}^{I-1}\\gamma_n)}` para :math:`j, l=0, 1,\\cdots, I-1, j\\neq l`.
        :math:`h_{i+1, I+2+j} = \\frac{1}{\\phi}` para :math:`i=1, 2, \\cdots, I` y :math:`j=0, 1, \\cdots, I-i`.
        :math:`h_{I+2+j, i+1} = \\frac{1}{\\phi}` para :math:`j=0, 1, \\cdots, I-1` y :math:`i=1, 2, \\cdots, I-j`.
        Las demás entradas son cero. Retorna un numpy array con las anteriores entradas.
    """
    matriz = np.zeros((2 * I + 1, 2 * I + 1))
    param_h3_comb = [(j, l) for j in range(I) for l in range(I) if j != l]
    for i in range(I + 1):
        matriz[i, i] = h1_[i]
    for j in range(I):
        matriz[I + j + 1, I + j + 1] = h2_[j]
    for i, (j, l) in enumerate(param_h3_comb):
        matriz[I + j + 1, I + l + 1] = h3_[i]
    for i in range(1, I + 1):
        for j in range(I - i + 1):
            matriz[i, I + 1 + j] = 1.0 / phi_
    for j in range(I):
        for i in range(1, I - j + 1):
            matriz[I + 1 + j, i] = 1.0 / phi_
    return matriz
def g_matrix(inverse_fisher_matrix, I):
    """
        Matriz requerida para calcular el error de estimación.
    :param inverse_fisher_matrix: Es la matriz inversa de la matriz de información
        de fisher de tamaño :math:`(2I+1)\\times(2I+1)` denotada por :math:`\\mathscr{H}^{-1}`
    :param int I: Período hasta el cual se dispone de información.
    :return:  Se denota por :math:`\\mathscr{G}` a la matriz resultado de tamaño :math:`I \\times I` y
        :math:`g_{i, j}` las entradas de esta matriz.
        Entonces,
        :math:`g_{j, l} = \\mathscr{H}^{-1}_{I+2+j, I+2+l}` para :math:`j, l=0, 1,\\cdots, I-1`.
        :math:`g_{j, I} = g_{I, j} = -\\sum_{m=0}^{I-1}\\mathscr{H}^{-1}_{I+2+j,I+2+m}` para :math:`j = 0, 1,\\cdots, I-1`.
        :math:`g_{I, I} = \\sum^{I-1}_{\\substack{0\\leq m \\leq I-1 \\\\ 0\\leq n \\leq I-1}}\\mathscr{H}^{-1}_{I+2+m,I+2+n}`.
    """
    h = inverse_fisher_matrix
    matrix = np.zeros((I + 1, I + 1))
    for j in range(I):
        for l in range(I):
            matrix[j, l] = h[I + 1 + j, I + 1 + l]
    for j in range(I):
        matrix[j, I] = matrix[I, j] = -sum([h[I + 1 + j, I + 1 + m] for m in range(I)])
    matrix[I, I] = sum([h[I + 1 + m, I + 1 + n] for m in range(I) for n in range(I)])
    return matrix
def xijs_deviation(mijs, phi, i, j):
    """
        Estima la desviación de :math:`X_{i,j}`.
    :param mijs: Matriz de tamaño :math:`I\\times I` donde la :math:`i, j`-ésima entrada corresponde con el parámetro
        :math:`m_{i, j}` (:math:`\\text{mij[i, j]} = m_{i, j}`).
    :param phi: Parámetro :math:`\\phi`.
    :param int i: Índice que indica el período de incurridos al cuál se le va a estimar la desviación de :math:`x_{i, j}`.
        Toma valores :math:`i=0, \cdots I`.
    :param int j: Índice que indica el período de reportados al cuál se le va a estimar la desviación de :math:`x_{i, j}`.
        Toma valores :math:`j=0, \cdots J`.
    :return: Cálculo de
        :math:`\\text{desviación de } X_{i, j} = \\phi m_{i, j}`
    """
    I, J = get_matrix_dimensions(mijs)
    if i+j > I:
        return np.sqrt(phi*mijs[i, j])
    else:
        return 0
def cij_deviation(phi_, gammas_, i, j, I, nus):
    """
        Estimador de la varianza del proceso.
    :param phi_: Parámetro :math:`\\phi`.
    :param gammas_: Estimación de parámetros :math:`\\gamma_j`. La lista gammas\_ debe
        contener las entradas computadas de los parámetros :math:`\\gamma_j` desde 0 hasta :math:`I`, donde
        se asume que la :math:`j`-ésima entrada de gammas\_ corresponde con :math:`\\gamma_j` (:math:`\\text{gammas_[j]} = \\gamma_j`).
    :param int i: Entero con valor entre 0 e :math:`I` (incluyendo 0 e :math:`I`) donde :math:`I` es el período hasta el que
        se tiene información.
    :param int j: Entero con valor entre 0 e :math:`I` (incluyendo 0 e :math:`I`) donde :math:`I` es el período hasta el que
        se tiene información.
    :param int I: Período hasta el cual se dispone de información.
    :param nus: Parámetros :math:`\\nu_i` correspondientes a la estimación a priori
        por parte de un experte de :math:`C_{i, J}`. La lista nus debe
        contener las entradas computadas de los parámetros :math:`\\nu_i` desde 0 hasta :math:`i+1`, donde
        se asume que la :math:`i`-ésima entrada de nus corresponde con :math:`\\nu_i` (:math:`\\text{nus[j]} = \\nu_j`).
    :return: Cuando :math:`i+j>I`, devuelve :math:`\\sum_{j>I-i}\\phi\\nu_i\\gamma_j`
        Cuando :math:`i+j\\leq I`, devuelve 0 porque es un parámetro ya conocido (no hay error).
    """
    if i+j <= I:
        return 0
    return np.sqrt(phi_ * nus[i] * sum(gammas_[I - i + 1: I + 1]))
def estimation_error(gammas_, g_matrix, i, j, I, nus, variances):
    """
        Estimador del error de estimación.
    :param gammas_: Estimación de parámetros :math:`\\gamma_j`. La lista gammas\_ debe
        contener las entradas computadas de los parámetros :math:`\\gamma_j` desde 0 hasta :math:`I`, donde
        se asume que la :math:`j`-ésima entrada de gammas\_ corresponde con :math:`\\gamma_j` (:math:`\\text{gammas_[j]} = \\gamma_j`).
    :param g_matrix: Matriz de tamaño :math:`I\\times I`.
    :param int i: Entero con valor entre 0 e :math:`I` (incluyendo 0 e :math:`I`) donde :math:`I` es el período hasta el que
        se tiene información.
    :param int j: Entero con valor entre 0 e :math:`I` (incluyendo 0 e :math:`I`) donde :math:`I` es el período hasta el que
        se tiene información.
    :param int I: Período hasta el cual se dispone de información.
    :param nus: Estimación de parámetros :math:`\\nu_i` estimados a priori por un experto. La lista nus debe
        contener las entradas computadas de los parámetros :math:`\\nu_i` desde 0 hasta :math:`i+1`, donde
        se asume que la :math:`i`-ésima entrada de nus corresponde con :math:`\\nu_i` (:math:`\\text{nus[j]} = \\nu_j`).
    :param variances: Estimación a priori de varianzas :math:`\\text{var}(\\nu_i)` , una por cada elemento de la lista nus. El :math:`i`-ésimo elemento
        de variances es interpretado como el error cometido por el experto al estimar la :math:`i`-ésima componente
        de la lista nus.
    :return: Cuando :math:`i+j>I`, devuelve :math:`\\left(\\sum_{j>I-i}\\gamma_j \\right)^2 \\text{var}(\\nu_i) + \\nu_i^2\\sum_{\\substack{j > I-i \\\\ l > I-i}}g_{j, l}`.
        Cuando :math:`i+j\\leq I`, devuelve 0 porque es un parámetro ya conocido (no hay error).
    """
    if i+j <= I:
        return 0
    s1 = variances[i] * sum(gammas_[I - i + 1: I + 1]) ** 2
    s2 = (nus[i] ** 2) * sum([g_matrix[k, l]
                              for k in range(j - i + 1, I + 1)
                              for l in range(j - i + 1, I + 1)])
    return np.sqrt(s1 + s2)
def estimate_ferguson(triangle, nus, variances):
    """
        Estima :math:`C_{i,j}` para :math:`i+j > I` con el modelo teórico Ferguson y los errores asociados a
        la estimación.
        Esta función primero estima los parámetros requeridos para estimar la matriz con los :math:`C_{i, j}` y los
        errores asociados. Finalmente, llama a la función estimate_ferguson (respectivamente a las que estiman el error)
        para estimar :math:`C_{i,j}`.
    :param np.ndarray triangle: Matriz de información. Véase la sección :ref:`preparación de los datos de entrada <triangle_format>`
        para más información.
    :param nus: Estimación de parámetros :math:`\\nu_i` estimados a priori por un experto. La lista nus debe
        contener las entradas computadas de los parámetros :math:`\\nu_i` desde 0 hasta :math:`I`, donde
        se asume que la :math:`i`-ésima entrada de nus corresponde con :math:`\\nu_i` (:math:`\\text{nus[j]} = \\nu_j`).
    :param variances: Estimación de varianzas :math:`\\text{var}_i` , una por cada elemento de la lista nus. El :math:`i`-ésimo elemento
        de variances es interpretado como el error cometido por el experto al estimar la :math:`i`-ésima componente
        de la lista nus.
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
    standard_triangle_check(triangle)
    assert I == J, "Aún no se soportan matrices rectangulares."
    assert len(nus) >= I, "Insuficientes estimaciones a priori para llevar a cabo la estimación con Ferguson"
    assert len(variances) >= I, "No se han especificado los suficientes errores para las estimaciones a priori"
    single_payments_matrix = single_payments_from_triangle(triangle)
    fjs = np.array([fj(triangle, j) for j in range(J)])
    cijs = np.array([[estimate_cij(triangle, fjs, i_, j_, nus) for j_ in range(J + 1)] for i_ in range(I + 1)])
    gammas = np.array([gamma(fjs, j, J) for j in range(J + 1)])
    calc_mus = np.array([calc_mu(triangle, fjs, i) for i in range(I + 1)])
    mijs = np.array([[mij(gammas, calc_mus, i_, j_) for j_ in range(J + 1)] for i_ in range(I + 1)])
    d_ = d(I)
    phi_ = phi(single_payments_matrix, d_, mijs)
    h1s = np.array([h1(phi_, gammas, calc_mus, i, I) for i in range(I + 1)])
    h2s = np.array([h2(phi_, gammas, calc_mus, j, I) for j in range(I + 1)])
    param_h3_comb = [(j, l) for j in range(J) for l in range(J) if j != l]
    h3s = np.array([h3(phi_, gammas, calc_mus, j_, l, I) for j_, l in param_h3_comb])
    try:
        fisher_matrix_ = fisher_matrix(phi_, h1s, h2s, h3s, I)
        fisher_matrix_inverse = np.linalg.inv(fisher_matrix_)
        g_matrix_ = g_matrix(fisher_matrix_inverse, I)
    except np.linalg.linalg.LinAlgError as err:
        if 'Singular matrix' in err.message:
            g_matrix_ = np.empty((I + 1, I + 1))
            g_matrix_[:] = np.nan
        else:
            raise
    deviations_cijs = np.array([[cij_deviation(phi_, gammas, i, j, I, nus) for j in range(J + 1)] for i in range(I + 1)])
    deviations_xijs = np.array([[xijs_deviation(mijs, phi_, i, j) for j in range(J+1)] for i in range(I+1)])
    est_errors = np.array([[estimation_error(gammas, g_matrix_, i, j, I, nus, variances) for j in range(J + 1)] for i in range(I + 1)])
    prediction_errors = np.array([[prediction_error(deviations_cijs, est_errors, i, j) for j in range(J + 1)] for i in range(I + 1)])
    estimation_error_norm = vco_matrix(est_errors, cijs)
    process_error_norm = vco_matrix(prediction_errors, cijs)
    salida = gen_standard_output(scale_factor*cijs, fjs, scale_factor*deviations_cijs, scale_factor*deviations_xijs, scale_factor*est_errors)
    return salida