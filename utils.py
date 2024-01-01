import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def channels_specs(file_loc):
    
    file = open(file_loc, 'r')

    raw_file = file.read()
    channel_name = raw_file.replace('\t', ' ').split("\n")

    file.close()
    
    return channel_name

def OrderEstimate_byChannels(Data, channels, max_order, min_order, leap_length):
    
    number_of_channels = len(channels)

    ctr = 0
    
    orders_mat = np.zeros((number_of_channels, number_of_channels))

    orders = np.arange(min_order, max_order, leap_length)
    
    for a, channel_a in enumerate(channels):

        for b, channel_b in enumerate(channels):

            if a != b:

                BICs = []

                x_t = Data[channel_a, :]
                y_t = Data[channel_b, :]

                for order in orders:

                    a_est_mul, b_est_mul = mulvar_AR_est(x_t, y_t, order, order)

                    x_t_rec_mat = x_t_recun_ab(a_est_mul, b_est_mul, x_t, y_t)
                    BICs.append(BIC_calc(x_t, x_t_rec_mat, order))
                    
                # plt.plot(BICs)
                # plt.show()
                
                ctr = ctr + 1
                print(int(ctr / (number_of_channels * (number_of_channels - 1)) * 100), "%", "Estimated Order is", orders[int(np.argmin(BICs))])

                orders_mat[a, b] = orders[int(np.argmin(BICs))]
                
    return orders_mat

def a_estimation_err(a_est, x_t):
    
    order = len(a_est)
    length = len(x_t)
    
    x_t_rec = np.zeros(length)
    x_t_rec[:order] = x_t[:order]

    for i in range(order, length):

        for j in range(order):

            x_t_rec[i] = x_t_rec[i] + a_est[j] * x_t[i - j - 1]

    return np.sum((x_t - x_t_rec) ** 2)

def x_t_recun(a_est, x_t):
    
    order = len(a_est)
    length = len(x_t)
    
    x_t_rec = np.zeros(length)
    x_t_rec[:order] = x_t[:order]

    for i in range(order, length):

        for j in range(order):

            x_t_rec[i] = x_t_rec[i] + a_est[j] * x_t[i - j - 1]

    return x_t_rec

def ab_estimation_err(a_est, b_est, x_t, y_t):
    
    a_order = len(a_est)
    b_order = len(b_est)
    
    length = len(x_t)
    
    x_t_rec = np.zeros(length)
    x_t_rec[:a_order] = x_t[:a_order]

    for i in range(a_order, length):

        for j in range(a_order):

            x_t_rec[i] = x_t_rec[i] + a_est[j] * x_t[i - j - 1]
            
        for j in range(b_order):

            x_t_rec[i] = x_t_rec[i] + b_est[j] * y_t[i - j - 1]

    return np.sum((x_t - x_t_rec) ** 2)

def x_t_recun_ab(a_est, b_est, x_t, y_t):
    
    a_order = len(a_est)
    b_order = len(b_est)
    
    length = len(x_t)
    
    x_t_rec = np.zeros(length)
    x_t_rec[:a_order] = x_t[:a_order]

    for i in range(a_order, length):

        for j in range(a_order):

            x_t_rec[i] = x_t_rec[i] + a_est[j] * x_t[i - j - 1]
            
        for j in range(b_order):

            x_t_rec[i] = x_t_rec[i] + b_est[j] * y_t[i - j - 1]

    return x_t_rec


def univar_AR_est(x_t, order):
    
    length = len(x_t)

    X_mat = np.zeros((length - order, order))
    X_vec = np.zeros((length - order))

    for i in range(length - order):

        X_mat[i, :] = x_t[i : i + order]
        X_vec[i] = x_t[i + order]

    return np.flip(np.linalg.pinv(X_mat) @ X_vec)

def mulvar_AR_est(x_t, y_t, a_order, b_order):
    
    length = len(x_t)

    X_mat = np.zeros((length - a_order, a_order + b_order))
    X_vec = np.zeros((length - a_order))

    for i in range(length - a_order):

        X_mat[i, : a_order] = x_t[i : i + a_order]
        X_mat[i, a_order : a_order + b_order] = y_t[i : i + b_order]
        
        X_vec[i] = x_t[i + a_order]

    coef_est = np.linalg.pinv(X_mat) @ X_vec
    
    a_est = np.flip(coef_est[:a_order])
    b_est = np.flip(coef_est[a_order : a_order + b_order])
    
    return a_est, b_est

def GC_calc(x_t, y_t, order):

    # it is much better to give access to orders to users!
    GC_val = []
    univar_error = []
    mulvar_error = []

    a_est_uni = univar_AR_est(x_t, order)

    a_order = order
    b_order = a_order

    a_est_mul, b_est_mul = mulvar_AR_est(x_t, y_t, a_order, b_order)

    univar_error.append(a_estimation_err(a_est_uni, x_t))
    mulvar_error.append(ab_estimation_err(a_est_mul, b_est_mul, x_t, y_t))
        
    GC_val.append(np.log(univar_error[-1] / mulvar_error[-1]))
        
    # print("Order is", order, "and Granger Causality is", np.log(a_estimation_err(a_est_uni, x_t) / ab_estimation_err(a_est_mul, b_est_mul, x_t, y_t)), "Univar Error is", a_estimation_err(a_est_uni, x_t), "and mulvar error is", ab_estimation_err(a_est_mul, b_est_mul, x_t, y_t))
        
    return GC_val, univar_error, mulvar_error

def AIC_calc(y, y_pred, k):
    
    # AIC = 2k + n * ln(mean sum of residuals)
    
    n = len(y)
    
    if len(y) != len(y_pred):
        
        print("Predicted values and real data doesn't have same length")
        
        return ''
    
    MSR = np.sum((y - y_pred) ** 2) / n
    
    return 2 * k + n * np.log(MSR)

def BIC_calc(y, y_pred, k):
    
    # BIC = k * ln(n) + n * ln(mean sum of residuals)
    
    n = len(y)
    
    if len(y) != len(y_pred):
        
        print("Predicted values and real data doesn't have same length")
        
        return ''
    
    MSR = np.sum((y - y_pred) ** 2) / n
    
    return k * np.log(n) + n * np.log(MSR)

def GrangerCausalityEstimator(Data, channels, window_length, overlap_ratio, orders_mat):
    
    garbage, N = Data.shape
    
    number_of_channels = len(channels)
    number_of_windows = int((N - window_length) / ((1 - overlap_ratio) * window_length)) + 1

    GC_values = np.zeros((number_of_windows, number_of_channels, number_of_channels))

    for win_step in range(number_of_windows):

        print("In Progress", win_step / number_of_windows * 100, "% ...")

        for i, channel_a in enumerate(channels):

            for j, channel_b in enumerate(channels):

                if i != j:

                    win_stp = int((win_step) * (1 - overlap_ratio) * window_length)
                    win_enp = win_stp + window_length

                    x_t = Data[channel_a, win_stp : win_enp]
                    y_t = Data[channel_b, win_stp : win_enp]

                    est_order = int(orders_mat[i, j])

                    tmp, tmp_err1, tmp_err2 = GC_calc(x_t, y_t, est_order)

                    GC_values[win_step, i, j] = tmp[0]
                    
    return GC_values
