import numpy as np
from scipy.ndimage import correlate, uniform_filter1d


def _size(size):
    """Lato della finestra (dispari) lungo frequenze e tempo: int o (f, t)."""
    sf, st = (size, size) if np.isscalar(size) else size
    if sf < 1 or st < 1 or sf % 2 == 0 or st % 2 == 0:
        raise ValueError("size deve essere dispari e >= 1 su entrambi gli assi")
    return int(sf), int(st)


def _box_sum(a, sf, st):
    """Somma di a su una finestra sf x st centrata, zero fuori dai bordi.
    Filtro separabile: costo indipendente dalla dimensione della finestra."""
    s = a
    if sf > 1:
        s = uniform_filter1d(s, sf, axis=0, mode="constant", cval=0.0) * sf
    if st > 1:
        s = uniform_filter1d(s, st, axis=1, mode="constant", cval=0.0) * st
    return s


def _box_count(shape, sf, st):
    """Numero di pixel della finestra sf x st che cadono dentro la matrice."""
    def count_1d(n, s):
        r = s // 2
        i = np.arange(n)
        return (np.minimum(i + r, n - 1) - np.maximum(i - r, 0) + 1).astype(np.float64)
    return np.outer(count_1d(shape[0], sf), count_1d(shape[1], st))


def _neighbour_sum(z, kernel, sf, st):
    """Restituisce (sum_j k_ij z_j, sum_j k_ij), con k_ii = 0.

    queen : rettangolo sf x st              (distanza di Chebyshev)
    rook  : rombo inscritto nel rettangolo  (distanza di Manhattan)
    time  : segmento lungo l'asse 1, lungo st
    freq  : segmento lungo l'asse 0, lungo sf
    array : kernel di pesi arbitrario (dimensioni dispari)
    """
    if isinstance(kernel, str) and kernel in ("queen", "time", "freq"):
        if kernel == "time":
            sf = 1
        elif kernel == "freq":
            st = 1
        if sf * st == 1:
            raise ValueError("finestra 1x1: nessun vicino")
        num = _box_sum(z, sf, st) - z                       # toglie il centro
        den = _box_count(z.shape, sf, st) - 1.0
        return num, den

    if isinstance(kernel, np.ndarray):
        K = kernel.astype(np.float64).copy()
        if K.shape[0] % 2 == 0 or K.shape[1] % 2 == 0:
            raise ValueError("il kernel deve avere dimensioni dispari")
    elif kernel == "rook":
        rf, rt = sf // 2, st // 2
        df = np.abs(np.arange(-rf, rf + 1))[:, None] / max(rf, 1)
        dt = np.abs(np.arange(-rt, rt + 1))[None, :] / max(rt, 1)
        K = ((df + dt) <= 1.0).astype(np.float64)
    else:
        raise ValueError(f"kernel non riconosciuto: {kernel}")

    K[K.shape[0] // 2, K.shape[1] // 2] = 0.0
    if not K.any():
        raise ValueError("kernel senza vicini")
    num = correlate(z, K, mode="constant", cval=0.0)
    den = correlate(np.ones_like(z), K, mode="constant", cval=0.0)
    return num, den


def LOCAL_MORAN(arr, kernel="queen", size=3, h=1.0, output=True):
    """Indice di Moran locale (Anselin 1995) su uno spettrogramma.

    Parametri
    ---------
    arr    : array 2D (frequenze x tempo), come in CUSUM.
    kernel : 'queen' | 'rook' | 'time' | 'freq' | np.ndarray dispari.
    size   : lato della finestra di vicinato, dispari. Intero (finestra
             quadrata) oppure tupla (size_freq, size_tempo).
             Esempi: size=3 -> queen 3x3 (8 vicini); size=11 -> queen 11x11
             (120 vicini); kernel='time', size=11 -> 5 vicini per lato
             lungo il tempo. Ignorato se kernel e' un array.
    h      : soglia su I_i per la maschera.
    output : False -> maschera (1 = pixel High-High con I_i > h)
             True  -> statistica I_i

    Definizioni
    -----------
    z_i = x_i - mean(x)
    m2  = sum(z^2) / n
    L_i = sum_j w_ij z_j          (lag spaziale, w_ij = k_ij / sum_j k_ij)
    I_i = z_i * L_i / m2
    Ai bordi la standardizzazione per riga usa solo i vicini esistenti.
    """
    x = np.asarray(arr, dtype=np.float64)
    sf, st = _size(size)

    z = x - x.mean()
    m2 = np.mean(z * z)

    num, den = _neighbour_sum(z, kernel, sf, st)
    L = num / den
    I = z * L / m2

    if output:
        return I

    # High-High: pixel sopra la media circondato da valori sopra la media
    # (z_i > 0 e I_i > h > 0 implicano L_i > 0)
    return ((z > 0) & (I > h)).astype(np.float64)
