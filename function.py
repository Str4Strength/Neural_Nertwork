import math
import warnings

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from matplotlib.cm import get_cmap



class StateSaver(): pass


def shape(x):
    static, dynamic = [x.shape.as_list(), tf.shape(x)]
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def exists(value):
    return value is not None


def empty(tensor):
    return shape(tensor) == []


def default(value, default):
    return value if exists(value) else default


def update(variable, value):
    with tf.control_dependencies([tf.assign(variable, value)]):
        return tf.identity(value)


def cast_tuple(value):
    return (value,) if not isinstance(value, tuple) else value


def cast_list(value):
    return [value,] if not isinstance(value, list) else value


def summ(tensor, mask=None, **kwargs):
    if exists(mask): tensor = tf.where(tf.equal(mask, 0), tf.zeros_like(tensor), tensor)
    return tf.reduce_sum(tensor, **kwargs)


def mean(tensor, mask=None, **kwargs):
    if exists(mask):     # binary mask only
        denominator = tf.reduce_sum(mask, **kwargs)
        return summ(tensor, mask=mask, **kwargs) / denominator
    return tf.reduce_mean(tensor, **kwargs)


def prod(tensor, mask=None, **kwargs):
    if exists(mask): tensor = tf.where(tf.equal(mask, 0), tf.ones_like(tensor), tensor)
    return tf.reduce_prod(tensor, **kwargs)


def refine_axis(axis, len_shape):
    if isinstance(axis, int): axis = (axis,)
    return [(x + len_shape if x < 0 else x) for x in axis]


def maxim(tensor, mask=None, **kwargs):
    if exists(mask):
        full_min = tf.reduce_min(tensor, axis=getattr(kwargs, 'axis', None), keepdims=True)
        tensor = tf.where(tf.equal(mask, 0), full_min * tf.ones_like(tensor), tensor)
    return tf.reduce_max(tensor, **kwargs)


def minim(tensor, mask=None, **kwargs):
    if exists(mask):
        full_max = tf.reduce_max(tensor, axis=getattr(kwargs, 'axis', None), keepdims=True)
        tensor = tf.where(tf.equal(mask, 0), full_max * tf.ones_like(tensor), tensor)
    return tf.reduce_min(tensor, **kwargs)


def reconstruct_mask(
        features,
        mask=None,
        axis=-1
        ):
    if not exists(mask): return

    full, is_neg = len(shape(mask)), int(axis < 0)
    front, back = axis, - axis - 1

    mask = tf.reduce_max(mask, axis=axis, keepdims=True)
    mask = tf.tile(mask, [1] * (front + is_neg * full)  + [features] + [1] * (back + (1 - is_neg) * full))
    return mask


def lp_norm(tensor, p=2, mask=None, axis=None):
    assert isinstance(p, int) or p == np.inf
    t_abs = tf.abs(tensor)
    if p == np.inf:
        t_max = maxim(t_abs, axis=axis, keepdims=True)
    else:
        t_max = tf.pow(summ(tf.pow(t_abs, p), mask=mask, axis=axis, keepdims=True), 1./p)
    #return tf.where(t_max > 0, tensor / t_max, tf.zeros_like(tensor))
    pos_max = tf.cast(t_max > 0, dtype=tensor.dtype)
    return tensor / (pos_max * t_max + (1 - pos_max))


def chunk_(tensor, chunks, axis=0):
    ts = shape(tensor)
    if axis < 0: axis += len(ts)
    total = ts[axis]
    fit_chunks = total // chunks
    cut = fit_chunks * chunks
    begins = [[0] * len(ts), [0] * axis + [cut] + [0] * (len(ts) - axis - 1)]
    sizes = list(zip(*[[s, s] for s in ts[:axis]] + [[cut, total - cut]] + [[s, s] for s in ts[axis+1:]]))
    main, remainder = tf.slice(tensor, begins[0], sizes[0]), tf.slice(tensor, begins[1], sizes[1])
    chunk = tf.split(main, fit_chunks, axis=axis).append(remainder)
    return chunk


def causal_padding(x, kernel_size, dilations, data_format=None):
    in_shape = x.shape
    left_pad = dilations[0] * (kernel_size[0] - 1)
    batch_rank = 1 if getattr(in_shape, 'ndims', None) is None else len(in_shape) - 2
    if data_format == 'channels_first':
        padding = [[0, 0]] * (batch_rank + 1) + [[left_pad, 0]]
    else:
        padding = [[0, 0]] * batch_rank + [[left_pad, 0], [0, 0]]
    return padding


def p_normalize(x, p=2, axis=-1, epsilon=1e-16):
    assert isinstance(p, int) or p==np.inf
    x_abs = tf.math.abs(x)
    if p == np.inf:
        p_norm = tf.maximum(tf.reduce_max(x_abs, axis=axis, keepdims=True), epsilon)
    else:
        p_norm = tf.math.pow(tf.maximum(tf.reduce_sum(tf.math.pow(x_abs, p), axis=axis, keepdims=True), epsilon), 1./p)
    return x / p_norm


def truncated_normal(x, mean=0.0, std=1., lowbd=-2., upbd=2.):
    def normal_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a-2*std) or (mean > b+2*std):
        warnings.warn('mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
                      'The distribution of values may be incorrect.', stacklevel=2)

    l = norm_cdf((lowbd - mean) / std)
    u = norm_cdf((upbd - mean) / std)

    uniform = tf.ones_like(x) / (2*(u - l))
    return uniform


def kaiming_uniform_init(fan_in):
    fan_in = tf.cast(fan_in, dtype=tf.float32)
    limit = tf.math.rsqrt(fan_in) if fan_in != 0 else 0.05
    return tf.random_uniform_initializer(minval=-limit, maxval=limit)


def spectral_normalize(weight, power_iteration_rounds=1, training=True, name=None):
    with tf.variable_scope(name, 'spectral_norm'):
        w_shape = shape(weight)
        w = tf.reshape(weight, [-1, w_shape[-1]])
        u_var = tf.get_variable('spectral_norm_u', shape=[shape(w)[0], 1],
                dtype=w.dtype, initializer=tf.random_normal_initializer,
                trainable=False)
        u = u_var

        for _ in range(power_iteration_rounds):
            v = tf.nn.l2_normalize(tf.einsum('ad,ao->do', w, u))
            u = tf.nn.l2_normalize(tf.einsum('ad,do->ao', w, v))

        if training:
            with tf.control_dependencies([u_var.assign(u, name='update_u')]):
                u = tf.identity(u)

        u = tf.stop_gradient(u)
        v = tf.stop_gradient(v)

        spectral_norm = tf.einsum('od,do->', tf.einsum('ao,ad->od', u, w), v)
        #spectral_norm.shape.assert_is_fully_defined()
        #spectral_norm.shape.assert_is_compatible_with([1,1])

        normalized = weight / tf.maximum(spectral_norm, 1e-8)
    return tf.reshape(normalized, w_shape)


_i0A = [
    -4.41534164647933937950E-18,
    3.33079451882223809783E-17,
    -2.43127984654795469359E-16,
    1.71539128555513303061E-15,
    -1.16853328779934516808E-14,
    7.67618549860493561688E-14,
    -4.85644678311192946090E-13,
    2.95505266312963983461E-12,
    -1.72682629144155570723E-11,
    9.67580903537323691224E-11,
    -5.18979560163526290666E-10,
    2.65982372468238665035E-9,
    -1.30002500998624804212E-8,
    6.04699502254191894932E-8,
    -2.67079385394061173391E-7,
    1.11738753912010371815E-6,
    -4.41673835845875056359E-6,
    1.64484480707288970893E-5,
    -5.75419501008210370398E-5,
    1.88502885095841655729E-4,
    -5.76375574538582365885E-4,
    1.63947561694133579842E-3,
    -4.32430999505057594430E-3,
    1.05464603945949983183E-2,
    -2.37374148058994688156E-2,
    4.93052842396707084878E-2,
    -9.49010970480476444210E-2,
    1.71620901522208775349E-1,
    -3.04682672343198398683E-1,
    6.76795274409476084995E-1
    ]

_i0B = [
    -7.23318048787475395456E-18,
    -4.83050448594418207126E-18,
    4.46562142029675999901E-17,
    3.46122286769746109310E-17,
    -2.82762398051658348494E-16,
    -3.42548561967721913462E-16,
    1.77256013305652638360E-15,
    3.81168066935262242075E-15,
    -9.55484669882830764870E-15,
    -4.15056934728722208663E-14,
    1.54008621752140982691E-14,
    3.85277838274214270114E-13,
    7.18012445138366623367E-13,
    -1.79417853150680611778E-12,
    -1.32158118404477131188E-11,
    -3.14991652796324136454E-11,
    1.18891471078464383424E-11,
    4.94060238822496958910E-10,
    3.39623202570838634515E-9,
    2.26666899049817806459E-8,
    2.04891858946906374183E-7,
    2.89137052083475648297E-6,
    6.88975834691682398426E-5,
    3.36911647825569408990E-3,
    8.04490411014108831608E-1
    ]

def _chbevl(x, vals):
    # x : B,     vals: n
    dtype = x.dtype
    x_past, x_prev = tf.ones_like(x), x
    x_coeffs = tf.stack([x_past, x_prev], axis=-1)     # B, len(vals)
    for n in range(2, len(vals)):
        x_curr = x * x_prev - x_past
        x_coeffs = tf.concat([x_coeffs, x_curr[Ellipsis, None]], axis=-1)
        x_past, x_prev = x_prev, x_curr
    b_last = tf.reduce_sum(x_coeffs * tf.cast(vals[::-1],
        dtype=dtype)[None], axis=-1)
    b_past = tf.reduce_sum(x_coeffs * tf.cast(vals[:-2][::-1]+[0,0],
        dtype=dtype)[None], axis=-1)
    return 0.5 * (b_last - b_past)

def _i0_1(tensor):
    return tf.math.exp(tensor) * _chbevl(tensor/2.0 - 2.0, _i0A)

def _i0_2(tensor):
    return tf.math.exp(tensor) * _chbevl(32.0/tensor - 2.0, _i0B) * tf.math.rsqrt(tensor)

def i0(tensor) :
    tf.math.abs(tensor)
    return tf.where(tf.math.less_equal(tensor, 8.0), _i0_1(tensor), _i0_2(tensor))


RP = [
        -8.99971225705559398224E8,
        4.52228297998194034323E11,
        -7.27494245221818276015E13,
        3.68295732863852883286E15,
        ]

RQ = [
        #/* 1.00000000000000000000E0, */
        6.20836478118054335476E2,
        2.56987256757748830383E5,
        8.35146791431949253037E7,
        2.21511595479792499675E10,
        4.74914122079991414898E12,
        7.84369607876235854894E14,
        8.95222336184627338078E16,
        5.32278620332680085395E18,
        ]

PP = [
        7.62125616208173112003E-4,
        7.31397056940917570436E-2,
        1.12719608129684925192E0,
        5.11207951146807644818E0,
        8.42404590141772420927E0,
        5.21451598682361504063E0,
        1.00000000000000000254E0,
        ]

PQ = [
        5.71323128072548699714E-4,
        6.88455908754495404082E-2,
        1.10514232634061696926E0,
        5.07386386128601488557E0,
        8.39985554327604159757E0,
        5.20982848682361821619E0,
        9.99999999999999997461E-1,
        ]

QP = [
        5.10862594750176621635E-2,
        4.98213872951233449420E0,
        7.58238284132545283818E1,
        3.66779609360150777800E2,
        7.10856304998926107277E2,
        5.97489612400613639965E2,
        2.11688757100572135698E2,
        2.52070205858023719784E1,
        ]

QQ = [
        #/* 1.00000000000000000000E0, */
        7.42373277035675149943E1,
        1.05644886038262816351E3,
        4.98641058337653607651E3,
        9.56231892404756170795E3,
        7.99704160447350683650E3,
        2.82619278517639096600E3,
        3.36093607810698293419E2,
        ]
"""
def polevl(tensor, coeff, N):
    assert len(coeff) >= N+1
    shp = [shape(tensor)[0], 1]
    nth = tf.ones_like(tensor)
    poly = tf.reshape(nth, shp)
    for n in range(1,N+1):
        nth *= tensor
        poly = tf.concat([poly, tf.reshape(nth, shp)], axis=-1)
    c = np.asarray(coeff[:N+1])[None]
    return tf.reduce_sum(poly * c, axis=-1)

def p1evl(tensor, coeff, N):
    assert len(coeff) >= N+1
    shp = [shape(tensor)[0], 1]
    nth = tf.ones_like(tensor)
    poly = tf.reshape(nth, shp)
    for n in range(1,N+1):
        nth *= tensor
        poly = tf.concat([poly, tf.reshape(nth, shp)], axis=-1)
    c = np.asarray(coeff[:N]+[1])[None]
    return tf.reduce_sum(poly * c, axis=-1)
"""
def poly_val(tensor, coeffs):
    # coeffs must be cN, cN-1, ..., c1, c0
    dtype=tensor.dtype
    B = shape(tensor)[0]
    C = shape(coeffs)[-1] # N+1
    coeffs = tf.conver_to_tensor(coeffs, dtype=dtype)
    poly = tf.math.cumprod(tf.broadcast_to(tensor, [B, C]), axis=-1, exclusive=True,
            reverse=True)
    poly *= tf.broadcast_to(coeffs, [B, C])
    return poly

def j1(tensor):
    # tensor shape [B]
    w = tf.math.abs(tensor)
    sign = tf.math.sign(tensor)

    def leq_5(x):
        z = tf.math.square(x)
        w = poly_val(z, RP[:4]) / p1evl(z, [1]+RQ[:8])
        w = w * x * (z-Z1) * (z-Z2)
        return (w)
    def g_5(x):
        w = 5. / x
        z = tf.math.square(w)
        p = poly_val(z, PP[:7]) / poly_val(z, PQ[:7])
        q = poly_val(z, QP[:8]) / poly_eval(z, [1]+QQ[:7])
        xn = x - tf.cast(2.35619449019234492885, dtype=x.dtype)
        p = p * tf.math.cos(xn) - w * q * tf.math.sin(xn)
        return p * tf.cast(7.9788456080286535587989E-1,
                dtype=x.dtype) * tf.math.rsqrt(x)

    result = tf.where(tf.less_equal(w, 5.0), leq_5(tensor), g_5(tensor))
    return sign * result

def get_lowpass_filter(
        n_taps,
        cutoff,
        width,
        fs=None,
        radial=False):
    # cutoff, fs tensor available
    #print(cutoff)
    assert n_taps >= 1
    assert len(shape(cutoff)) == 1
    if n_taps == 1: return None
    if fs is None: fs = 2
    nyq = .5 * tf.cast(fs, tf.float32) # B,
    cutoff = tf.cast(cutoff, tf.float32)

    # attenuation, beta, kaiser_window
    alpha = .5 * (n_taps - 1) # []
    m = tf.range(n_taps, dtype=tf.float32) - alpha # n_taps
    atten = (2.285 * (n_taps - 1) * np.pi * tf.cast(width, tf.float32) / nyq) # B,
    atten_minus_21 = atten - 21 # B,
    beta = tf.where(tf.math.less_equal(atten_minus_21, 0.0), tf.zeros_like(atten),
            tf.where(tf.math.greater(atten, 50), 0.1102 * (atten - 8.7),
                0.5842 * atten_minus_21**0.4 + 0.07886 * atten_minus_21)) # B,
    window = i0(beta[Ellipsis, None] * tf.math.sqrt(
        1.0 - tf.math.square(m / alpha))[None]) / i0(beta[Ellipsis, None]) # B, n_taps

    if radial:
        x = m[None] / fs[Ellipsis, None]
        x_square = tf.math.square(x)
        pi_r = np.pi * tf.math.sqrt(x_square[:, None] + x_square[Ellipsis, None])
        j1_inputs = 2 * tf.broadcast(cutoff,[shape(cutoff)[0], 1, 1]) * pi_r
        # TODO bessel_j1 into batch supported tensorflow function
        #f = tf.math.special.bessel_j1(j1_inputs) / pi_r
        f = j1(j1_inputs) / pi_r
        f = f[:, None, None] * (window[Ellipsis, None] * window[:, None])
        # B, n_taps, n_taps
    else:
        cutoff = tf.broadcast_to(cutoff, [shape(cutoff)[0]]) / nyq
        bands = tf.stack([tf.zeros_like(cutoff), cutoff], axis=-1)
        pi_bands_m = (np.pi * bands)[Ellipsis, None] * m[None, None]
        left_right = bands[Ellipsis, None] * tf.where(tf.equal(pi_bands_m, 0.0),
                tf.ones_like(pi_bands_m),
                tf.math.sin(pi_bands_m) / pi_bands_m)     # B, 2, n_taps
        left_right *= np.asarray([-1, 1])[None, :, None]
        f = tf.reduce_sum(left_right, axis=-2)     # B, n_taps
        f *= window
        # B, n_taps

    f /= tf.reduce_sum(f)
    #print(shape(f))
    return f


def upfirdn_2d(
        tensor,
        kernel_hw=None,
        kernel_h=None,
        kernel_w=None,
        uh=1,
        uw=1,
        dh=1,
        dw=1,
        phl=0,
        phr=0,
        pwl=0,
        pwr=0,
        gain_hw=1.0,
        gain_h=1.0,
        gain_w=1.0,
        flip_filter_hw=False,
        flip_filter_h=False,
        flip_filter_w=False,
        mask=None,
        ):
    assert len(shape(tensor)) == 4
    #assert (kernel_hw is not None or kernel_h is not None or kernel_w is not None)
    dtype = tensor.dtype
    B, H, W, C = shape(tensor)

    tensor = tf.pad(tensor[:, :, None, :, None, :],
            [[0,0], [0,0], [0,uh-1], [0,0], [0,uw-1], [0,0]])
    tensor = tf.reshape(tensor, [B, H*uh, W*uw, C])
    tensor = tf.pad(tensor,
            [[0,0], [tf.math.maximum(phl,0), tf.math.maximum(phr,0)],
                [tf.math.maximum(pwl,0), tf.math.maximum(pwr,0)], [0,0]])
    tensor = tensor[:, tf.math.maximum(-phl,0):shape(tensor)[1]-tf.math.maximum(-phr,0),
            tf.math.maximum(-pwl,0):shape(tensor)[2]-tf.math.maximum(-pwr,0), :]
    if mask is not None:
        mask = tf.tile(mask[:,:,None,:,None,:], [1,1,uh,1,uw,1])
        mask = tf.reshape(mask, [B,H*uh,W*uw,C])
        mask = tf.pad(mask,
                [[0,0],[tf.math.maximum(phl,0),tf.math.maximum(phr,0)],
                    [tf.math.maximum(pwl,0),tf.math.maximum(pwr,0)],[0,0]])
        mask = mask[:,tf.math.maximum(-phl,0):shape(tensor)[1]-tf.math.maximum(-phr,0),
                tf.math.maximum(-pwl,0):shape(tensor)[2]-tf.math.maximum(-pwr,0),:]
    if kernel_hw is None and kernel_h is None and kernel_w is None: return tensor, mask

    tensor = tf.transpose(tensor, [3,1,2,0]) # C, phl+H*uh+phr, pwl+W*uw+pwr, B

    if kernel_hw is not None: # B, kH, kW
        _, kH, kW = shape(kernel_hw)
        kernel_hw = kernel_hw * (gain_hw ** (len(shape(kernel_hw)) / 2.0))
        if not flip_filter_hw: kernel_hw = kernel_hw[:, ::-1, ::-1]
        weight = tf.constant(tf.transpose(kernel, [1,2,0])[:, :, None],
                dtype=dtype) # kH, kW, 1, B
        tensor = tf.nn.conv2d(tensor, weight, strides=[1,1,1,1], padding='VALID')

    else:
        kH, kW = 1, 1
        if kernel_h is not None: # B, kH
            _, kH = shape(kernel_h)
            kernel_h = kernel_h * (gain_h ** (len(shape(kernel_h)) / 2.0))
            if not flip_filter_h: kernel_h = kernel_h[:, ::-1]
            # kH, 1, 1, B
            weight_h = tf.transpose(kernel_h, [1,0])[:, None, None]
            tensor = tf.nn.conv2d(tensor, weight_h, strides=[1,1,1,1], padding='VALID')

        if kernel_w is not None: # B, kW
            _, kW = shape(kernel_w)
            kernel_w = kernel_w * (gain_w ** (len(shape(kernel_w)) / 2.0))
            if not flip_filter_w: kernel_w = kernel_w[:, ::-1]
            weight_w = tf.transpose(kernel_w, [1,0])[None, :, None]
            # 1, kW, 1, B
            tensor = tf.nn.conv2d(tensor, weight_w, strides=[1,1,1,1], padding='VALID')
    if mask is not None:
        mask = tf.nn.max_pool2d(mask, ksize=[kH, kW], strides=1, padding='VALID')

    tensor = tf.transpose(tensor, [3,1,2,0])[:, ::dh, ::dw, :]
    if mask is not None:
        mask = tf.reshape(mask[:, ::dh, ::dw, :], shape(tensor))
        tensor *= mask

    return tensor, mask


def FIR_lpf(
        sampling_rate,
        filter_size,
        up_factor=None,
        down_factor=None,
        radial=False):
    if up_factor is not None and down_factor is not None:
        fs = sampling_rate * down_factor
        rate = fs//down_factor
        taps = filter_size * down_factor
    elif up_factor is not None:
        fs = sampling_rate * up_factor
        rate = tf.identity(fs)
        taps = filter_size * up_factor
    elif down_factor is not None:
        fs = sampling_rate
        rate = sampling_rate * down_factor
        taps = filter_size * down_factor
    else:
        return
    stopband, cutoff = tf.math.floor(.5*rate), tf.math.ceil(.25*rate)
    width = 2*(stopband - cutoff)

    FIR = get_lowpass_filter(n_taps=taps, cutoff=cutoff, width=width,
            fs=fs, radial=radial)

    return FIR

def upfirdn_1d(
        tensor,
        kernel=None,
        u=1,
        d=1,
        pl=0,
        pr=0,
        gain=1.0,
        flip_filter=False,
        mask=None
        ):
    assert len(shape(tensor)) == 3
    dtype = tensor.dtype
    B, L, C = shape(tensor)

    #gname = tf.get_default_graph().get_name_scope()
    #with tf.control_dependencies([
    #    tf_print(f'tensor_{gname}', tensor, color='green'),
    #    ]):
    #    with tf.control_dependencies([
    #        tf_print(f'mask_{gname}', mask, color='green'),
    #        ]):
    #        with tf.control_dependencies([
    #            tf_print(f'u_{gname}', u, color='green'),
    #            ]):
    #            tensor = tf.identity(tensor)

    tensor = tf.pad(tensor[:, :, None, :], [[0,0], [0,0], [0,u-1], [0,0]])
    tensor = tf.reshape(tensor, [B, L*u, C])
    tensor = tf.pad(tensor,
            [[0,0],[tf.math.maximum(pl,0), tf.math.maximum(pr,0)],[0,0]])
    tensor = tensor[:,
        tf.math.maximum(-pl,0):shape(tensor)[1]-tf.math.maximum(-pr,0), :]
    if mask is not None:
        #with tf.control_dependencies([
        #    tf_print('tensor', tensor, color='green'),
        #    tf_print('mask', mask, color='green'),
        #    tf_print('u', u, color='green'),
        #    ]):
        #    mask = tf.identity(mask)
        mask = tf.tile(mask[:,:,None,:], [1,1,u,1])
        mask = tf.reshape(mask, [B, L*u, C])
        mask = tf.pad(mask,
            [[0,0],[tf.math.maximum(pl,0), tf.math.maximum(pr,0)],[0,0]])
        mask = mask[:,
            tf.math.maximum(-pl,0):shape(mask)[1]-tf.math.maximum(-pr,0), :]
    if kernel is None: return tensor, mask
    #with tf.control_dependencies([
    #    tf_print('tensor', tensor),
    #    tf_print('mask', mask),
    #    ]):
    #    tensor = tf.identity(tensor)
    tensor = tf.transpose(tensor, [2,1,0]) # C, pl+L*u+pr, B

    _, K = shape(kernel)
    kernel = kernel * (gain ** (len(shape(kernel)) / 2.0))
    if not flip_filter: kernel = kernel[:, ::-1]
    # k, 1, B
    weight = tf.transpose(kernel, [1,0])[:, None]
    tensor = tf.nn.conv1d(tensor, weight, stride=1, padding='VALID')
    if mask is not None:
        mask = tf.nn.max_pool1d(mask, K, strides=1, padding='VALID')

    tensor = tf.transpose(tensor, [2,1,0])[:, ::d, :]
    if mask is not None:
        mask = tf.reshape(mask[:, ::d, :], shape(tensor))
        tensor *= mask

    return tensor, mask


def pos_enc_1d(
        tensor,
        mask=None
        ):
    dtype = tensor.dtype
    B, L, C = shape(tensor)
    if C%2 != 0: return tf.zeros_like(tensor)

    div = tf.math.exp(tf.range(start=0, limit=C, delta=2, dtype=dtype)*
        -(tf.math.log(10000.)/C))

    d_pos = tf.range(L, dtype=dtype)[:, None] * div[None]

    pe = tf.stack([tf.math.sin(d_pos), tf.math.cos(d_pos)], axis=-1) # L, C//2, 2
    abs_pe = tf.reshape(pe, [L, C])
    pos_enc = abs_pe[None] # 1, L, C
    if mask is not None: pos_enc*=mask
    return pos_enc


def pos_enc_2d(
        tensor,
        mask=None
        ):
    dtype = tensor.dtype
    B, H, W, C = shape(tensor)
    if C%4 != 0: return tf.zeros_like(tensor)

    div = tf.math.exp(tf.range(start=0, limit=C//2, delta=2, dtype=dtype)*
        -(tf.math.log(10000.)/(C//2)))

    d_pos_h = tf.range(H, dtype=dtype)[:, None] * div[None]
    d_pos_w = tf.range(W, dtype=dtype)[:, None] * div[None]

    pe_h = tf.stack([tf.math.sin(d_pos_h), tf.math.cos(d_pos_h)], axis=-1) # H, C//4, 2
    pe_w = tf.stack([tf.math.sin(d_pos_w), tf.math.cos(d_pos_w)], axis=-1) # W, C//4, 2
    abs_pe = tf.concat([tf.tile(pe_w[None], [H, 1, 1, 1]),
        tf.tile(pe_h[:, None], [1, W, 1, 1])], axis=-1)
    abs_pe = tf.reshape(abs_pe, [H, W, C])
    pos_enc = abs_pe[None] # 1, H, W, C
    if mask is not None: pos_enc*=mask
    return pos_enc


def rel_pos_emb(
        l_q,
        l_k,
        depth,
        max_rel_pos,
        cache=False,
        trainable=True,
        scope='rel_pos_emb',
        ):
    with tf.variable_scope(scope):
        # rel_pos_mat
        if not cache:
            range_q = tf.range(l_q)
            range_k = tf.range(l_k)
            distance_mat = range_k[None] - range_q[:, None]
        else:
            distance_mat = tf.range(-l_k+1, 1, 1)[None]
        distance_mat_clipped = tf.clip_by_value(distance_mat, -max_rel_pos, max_rel_pos)
        rel_pos_mat = distance_mat_clipped + max_rel_pos

        vocab_size = max_rel_pos * 2 + 1
        embedding_table = tf.get_variable("embeddings", [vocab_size, depth], initializer=tf.zeros_initializer,
                trainable=False)
        embeddings = tf.gather(embedding_table, rel_pos_mat)
        return embeddings


def group(
        tensor,
        group_size,
        axis,
        mask=None
        ):
    shp = shape(tensor)
    if axis < 0: axis += len(shp)

    reshp_0 = shp[:axis] + [group_size, shp[axis]//group_size] + shp[axis + 1:]
    trpose = [0, axis, *list(range(1, axis)), *list(range(axis + 1, len(shp) + 1))]
    reshp_1 = [shp[0] * group_size] + [shp[axis]//group_size if axis==n else d
            for n, d in enumerate(shp)][1:]

    output = tf.reshape(tf.transpose(tf.reshape(tensor, reshp_0), trpose), reshp_1)
    if mask is not None:
        mask = tf.reshape(tf.transpose(tf.reshape(mask, reshp_0), trpose), reshp_1)
        output *= mask

    return output, mask


def ungroup(
        tensor,
        group_size,
        axis,
        mask=None
        ):
    shp = shape(tensor)
    if axis < 0: axis += len(shp)

    reshp_0 = [shp[0]//group_size, group_size] + shp[1:]
    trpose = [0, *list(range(2, axis + 1)), 1, *list(range(axis + 1, len(shp) + 1))]
    reshp_1 = [shp[0]//group_size, *shp[1:axis], shp[axis]*group_size, *shp[axis+1:]]

    output = tf.reshape(tf.transpose(tf.reshape(tensor, reshp_0), trpose), reshp_1)
    if mask is not None:
        mask = tf.reshape(tf.transpose(tf.reshape(mask, reshp_0), trpose), reshp_1)
        output *= mask

    return output, mask

