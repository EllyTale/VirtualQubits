import numpy as np


def resample_x_fit(x):
    if len(x) < 1000:
        return np.linspace(np.min(x), np.max(x), 1001)
    else:
        return x


def resample_fit(x, y):
    from scipy.signal import savgol_filter
    if len(x) > 1000:
        factor = len(x) // 1000
        print(len(x[::factor]))
        return x[::factor], y[::factor]
    else:
        return x, y


def sin_fit(xdata, ydata):
    x, y = resample_fit(xdata, ydata)
    f = np.fft.fftfreq(len(x), (x[1] - x[0]))
    ft = abs(np.fft.fft(y))
    freq = abs(f[np.argmax(ft[1:]) + 1])
    offset = np.mean(y)
    amp = np.max(y) - offset
    estimated_params = np.array([freq, 1, 0, 0])
    # print(estimated_params)

    def model_func(t, f, a, p, c): return a * np.sin(2 * np.pi * f * t + p) + c
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(model_func, x, y, p0=estimated_params)
    fittered_func = model_func(resample_x_fit(x), *popt)
    parameters = {'frequency': popt[0], 'amplitude': popt[1], 'phase': popt[2], 'offset': 0}
    return (resample_x_fit(x), fittered_func), parameters


def exp_sin_fit(xdata, ydata, gamma=0):
    x, y = resample_fit(xdata, ydata)
    f = np.fft.fftfreq(len(x), (x[1] - x[0]))
    ft = abs(np.fft.fft(y))
    freq = abs(f[np.argmax(ft[1:]) + 1])
    offset = np.mean(y)
    amp = np.max(y) - offset
    estimated_params = np.array([freq, 1, 0, 0, gamma])
    # print(estimated_params)

    def model_func(t, f, a, p, c, g): return a * np.sin(2 * np.pi * f * t + p) * np.exp(- g * t) + c
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(model_func, x, y, p0=estimated_params)
    fittered_func = model_func(resample_x_fit(x), *popt)
    parameters = {'frequency': popt[0], 'amplitude': popt[1], 'phase': popt[2], 'offset': popt[3], 'gamma': popt[4]}
    return (resample_x_fit(x), fittered_func), parameters



# def rabi_fit(xdata, ydata, param: int = 1, prominence=0.9):
#     from scipy.optimize import curve_fit
#     from scipy.signal import find_peaks
#
#     def sin_fun(t, f, a, phase, offset):
#         return a * np.cos(2 * np.pi * f * t + phase) + offset
#         # return a * np.sin(2 * np.pi * f * t + phase) + offset
#
#     peaks, properties = find_peaks(ydata[::param], prominence=prominence)
#     print(peaks)
#
#     if len(peaks) >= 2:
#         f_r = 1 / (xdata[::param][peaks[1]] - xdata[::param][peaks[0]])
#     else:
#         # f_r = 1 / xdata[-1]
#         raise ValueError("Could not find any peaks!")
#     print(f_r)
#     popt, _ = curve_fit(sin_fun, xdata[::param], ydata[::param], p0=[f_r, 0.5, 0, 0.5])
#     fitting_result = sin_fun(xdata, *popt)
#     parameters = {'frequency': popt[0], 'amplitude': popt[1], 'phase': popt[2], 'offset': 0}
#
#     return (xdata, fitting_result), parameters





