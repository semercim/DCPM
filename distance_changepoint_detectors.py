import numpy as np

from numpy import linalg as la
from histograms import SlidingWindow
from scipy.stats import chi2
import matplotlib.pyplot as plt

_author__ = 'murat.semerci@boun.edu.tr'


class ODBOD(object):
    """The class for Online Distance Based Outlier Detector, holds the optimal Mahalanobis Matrix
       The log det divergence function is used
    """

    def __init__(self, stat_type=None, dim_input=1, num_predecessor=5, lambda_p=1.0, beta_p=1.0,
                 initial_mahalanobis_matrix=None):
        # ODBOD(dim_input, stat_type,  dim_input=1, num_predecessor=5, lambda_p=1, initial_mahalanobis_matrix=None)
        # Params:
        # stat_type is the STATS to be used, required for window creation
        # dim_input is the dimension of input space, d
        # num_predecessor is the number of neighbors, k
        # lambda_p is the penalty weight for logdet divergence, old
        # beta_p is the penalty weight for logdet divergence, identity
        # initial_mahalanobis_matrix is the 0th Mahalanobis matrix
        #
        # sets mahalanobis_matrix, the dim_input x dim_input Mahalanobis Matrix
        # sets num_predecessor, the number of predecessor
        # sets lambda_p, the penalty weight for logdet divergence regularization, old matrix
        # sets beta_p, the penalty weight for logdet divergence regularization, identity
        # sets old_mahalanobis_matrix, the previous dim_input x dim_input Mahalanobis Matrix
        # sets dim_input, the input space dimensions
        # sets data, the data matrix that holds last k+1 instances

        if stat_type is None:
            print("The STATS type has to be entered in the ODBOD constructor! Exiting the application!")
            exit()

        self.lambda_p = float(lambda_p)
        self.beta_p = float(beta_p)
        self.dim_input = dim_input
        self.stat_type = stat_type
        self.num_predecessor = num_predecessor
        self.identity = np.identity(self.dim_input)
        self.mahalanobis_matrix = self.identity.copy()
        self.mahalanobis_matrix_inv = la.inv(self.mahalanobis_matrix)
        self.current_distances_sum = 0.0
        self.current_error_value = 0.0

        # below are the optional parameters
        self.save_distances = False
        self.result_filename = "Distances.txt"
        self.result_file = None
        self.print_verbose = False

        if initial_mahalanobis_matrix is not None:
            self.old_mahalanobis_matrix = initial_mahalanobis_matrix.copy()
            try:
                self.old_mahalanobis_matrix_inv = la.inv(self.old_mahalanobis_matrix)
            except Exception as exc:
                print(exc)
                self.old_mahalanobis_matrix = self.identity.copy()
                self.old_mahalanobis_matrix_inv = self.identity.copy()
        else:
            self.old_mahalanobis_matrix = self.identity.copy()
            self.old_mahalanobis_matrix_inv = self.identity.copy()

        self.data = np.zeros(shape=[self.dim_input, 0])
        self.y = np.zeros(shape=[1, 0])
        self.t = 0

    def create_odbodwindows(self):
        self.odbod_window_distance = ODBODWindow(self.stat_type)
        self.odbod_window_distance.set_info_texts('Time vs Distance', 'Time', 'Distance')
        self.odbod_window_distance.create()

    def set_mahalanobis_matrix(self, mahalanobis_matrix):
        # sets the Mahalanobis Matrix
        self.mahalanobis_matrix = mahalanobis_matrix.copy()

    def get_mahalanobis_matrix(self):
        # returns the Mahalanobis Matrix
        return self.mahalanobis_matrix.copy()

    def get_mahalanobis_matrix_inv(self):
        # returns the Mahalanobis Matrix
        return self.mahalanobis_matrix_inv.copy()

    def set_lambda_p(self, lambda_p):
        # sets the lambda coefficient
        self.lambda_p = lambda_p

    def get_lambda_p(self):
        # returns the lambda coefficient
        return self.lambda_p

    def set_beta_p(self, beta_p):
        # sets the lambda coefficient
        self.beta_p = beta_p

    def get_beta_p(self):
        # returns the lambda coefficient
        return self.beta_p

    def set_result_save(self, filename=None):
        self.save_distances = True
        if filename:
            self.result_filename = filename
            self.result_file = open(self.result_filename, "w")

    def get_time(self):
        # returns the time
        return self.t

    def print_verbosely(self, print_verbose=True):
        # print the time verbosely
        self.print_verbose = print_verbose

    def add_vector_to_window(self, x):  # added for 2-Mahalanobis systems
        # store the previous k instances
        if self.data.shape[1] <= self.num_predecessor:
            self.data = np.concatenate((self.data, np.matrix(x).T), axis=1)
        else:
            self.data = np.concatenate((self.data[:, -self.num_predecessor:], np.matrix(x).T), axis=1)

    def frobenius_normalize(self):
        fro_norm = la.norm(self.mahalanobis_matrix, 'fro')
        self.mahalanobis_matrix *= (float(self.dim_input)/fro_norm)
        self.mahalanobis_matrix_inv *= (fro_norm /float(self.dim_input))

    def calculate_distance(self, x1, x2):
        # returns squared projected Euclidean distance
        return (x1 - x2).transpose() * self.mahalanobis_matrix * (x1 - x2)

    def calculate_distances_sum(self):
        # calculates the objective function in the original formulation (sum of k precedessor distance)
        # current instance
        last_instance = self.data[:, -1]
        distances_sum = 0.0
        if self.data.shape[1] > self.num_predecessor:
            # there are more than or equal to k predecessor
            for k in range(-self.num_predecessor - 1, -1):
                distances_sum += self.calculate_distance(self.data[:, k], last_instance)
        else:
            # there are less than k predecessor
            for k in range(-self.data.shape[1], -1):
                distances_sum += self.calculate_distance(self.data[:, k], last_instance)
        return float(distances_sum)

    def log_det_divergence_pre(self):
        (sign, log_det) = la.slogdet(self.mahalanobis_matrix * self.old_mahalanobis_matrix_inv)
        return np.trace(self.mahalanobis_matrix * self.old_mahalanobis_matrix_inv) - log_det - self.dim_input

    def log_det_divergence_identity(self):
        (sign, log_det) = la.slogdet(self.mahalanobis_matrix)
        return np.trace(self.mahalanobis_matrix) - log_det - self.dim_input

    def calculate_error_value(self):
        # calculates the value of the error function
        return self.calculate_distances_sum() + self.lambda_p * self.log_det_divergence_pre() + \
               self.beta_p * self.log_det_divergence_identity()

    def calculate_gradient(self):
        # calculates the gradient for the error function
        last_instance = self.data[:, -1]
        grad = np.zeros((self.dim_input, self.dim_input))
        if self.data.shape[1] > self.num_predecessor:
            # there are more than or equal to k predecessor
            for k in range(-self.num_predecessor - 1, -1):
                difference_vector = self.data[:, k] - last_instance
                grad += np.multiply(difference_vector, difference_vector.transpose())
        else:
            # there are less than k predecessor
            for k in range(-self.data.shape[1], -1):
                difference_vector = self.data[:, k] - last_instance
                grad += np.multiply(difference_vector, difference_vector.transpose())
        return grad.copy()

    def process(self, x):
        self.t += 1
        if self.print_verbose:
            print("Data received at t=", self.t, ", ", x)
        # store the previous k instances
        if self.data.shape[1] <= self.num_predecessor:
            self.data = np.concatenate((self.data, np.matrix(x).T), axis=1)
        else:
            self.data = np.concatenate((self.data[:, -self.num_predecessor:], np.matrix(x).T), axis=1)

        self.current_error_value = self.calculate_error_value()
        self.current_distances_sum = self.calculate_distances_sum()
        initial_mahalanobis_matrix = self.get_mahalanobis_matrix()
        initial_mahalanobis_matrix_inv = self.get_mahalanobis_matrix_inv()
        self.mahalanobis_matrix_inv = float(self.lambda_p) / float(
            self.lambda_p + self.beta_p) * self.old_mahalanobis_matrix_inv + float(self.beta_p) / float(
            self.lambda_p + self.beta_p) * self.identity.copy() + 1.0 / float(
            self.lambda_p + self.beta_p) * self.calculate_gradient()
        self.mahalanobis_matrix = la.inv(np.matrix(self.mahalanobis_matrix_inv))
        self.old_mahalanobis_matrix_inv = initial_mahalanobis_matrix_inv
        self.old_mahalanobis_matrix = initial_mahalanobis_matrix

        if self.data.shape[1] <= self.num_predecessor:
            self.y = np.concatenate((self.y, np.matrix(self.current_distances_sum)), axis=1)
        else:
            self.y = np.concatenate((self.y[:, -self.num_predecessor:],
                                     np.matrix(self.current_distances_sum)), axis=1)
        if self.save_distances:
            self.result_file.write(str(self.current_distances_sum) + '\n')

    def update_windows(self):
        self.odbod_window_distance.update(self.current_distances_sum, loss_msg='Distances sum:',
                                          print_verbose=self.print_verbose)

    def __del__(self):
        if self.save_distances:
            self.result_file.close()


class ODBODWindow(SlidingWindow):
    def __init__(self, stat_type, width=800, height=250, dpi=96, y_tick_pos_max=100, y_tick_labels_interval=50.0):
        super(ODBODWindow, self).__init__('ODBODWindow', stat_type, width, height, dpi)
        self.title = 'Time vs Distance'
        self.x_label = 'Time Frame (sec.)'
        self.y_label = 'Distance'
        # y_ticks
        self.y_tick_pos = np.arange(0, y_tick_pos_max, y_tick_labels_interval)
        self.y_tick_labels = np.arange(0, y_tick_pos_max, y_tick_labels_interval)
        # data:
        self.loss = np.zeros(self.WINDOW_LENGTH)

    def set_info_texts(self, title='Title', x_label='x', y_label='y'):
        self.title = title
        self.x_label = x_label
        self.y_label = y_label

    def create(self):
        super(ODBODWindow, self).create()
        self.draw()

    def update(self, loss, loss_msg='Loss', print_verbose=True):
        if print_verbose:
            print(loss_msg, loss)
        # update data
        self.loss[:-1] = self.loss[1:]
        self.loss[-1] = loss
        # draw
        x_val = np.arange(0, self.WINDOW_LENGTH)
        self.ax.clear()
        self.ax.plot(x_val, self.loss, '-b')
        super(ODBODWindow, self).draw()


class CustomODBODHistogram(SlidingWindow):
    def __init__(self, width=800, height=500, dpi=96, headers=None, name='CustomODBODHistogram'):
        if headers is None:
            print("Headers can not be NONE! Exiting")
            exit()
        else:
            self.headers = headers
            self.NUM_HEADERS = len(self.headers)
            height = 200 + 10*self.NUM_HEADERS
        super(CustomODBODHistogram, self).__init__(name, [name], width, height, dpi)
        plt.ion()
        # titles, labels:
        self.title = 'Magnitude of Features'
        self.x_label = 'Time Frame (sec.)'
        self.y_label = 'Feature Type'
        # x - y axis labels
        self.y, self.x = np.mgrid[slice(0, self.NUM_HEADERS + 1, 1),
                                  slice(0, self.WINDOW_LENGTH + 1, 1)]
        self.histogram = np.zeros((self.NUM_HEADERS, self.WINDOW_LENGTH))
        # y-ticks
        self.y_tick_pos = np.arange(self.NUM_HEADERS) + 0.5
        self.y_tick_labels = self.headers
        self.max_value = 50
        # Initialize graph:
        super(CustomODBODHistogram, self).create()

    def set_max_value(self, max_value):
        self.max_value = max_value

    def handle_message(self, data_vector):
        self.update(data_vector)

    def update(self, data_vector):
        # shift window and append new data
        self.histogram[:, :-1] = self.histogram[:, 1:]
        self.histogram[:, -1] = data_vector

        # plot histogram
        self.ax.pcolormesh(self.x, self.y, self.histogram, cmap=plt.cm.Greys, vmin=0, vmax=self.max_value)
        super(CustomODBODHistogram, self).draw()


class ODBODTHRESHOLD(ODBOD):
    def __init__(self, stat_type=None, dim_input=1, num_predecessor=5, threshold=None, lambda_p=1.0, beta_p=1.0,
                 initial_mahalanobis_matrix=None, max_threshold=None):
        super(ODBODTHRESHOLD, self).__init__(stat_type, dim_input, num_predecessor, lambda_p, beta_p,
                                             initial_mahalanobis_matrix)
        self.print_verbosely(False)
        if threshold is None:
            self.threshold = 0.95
        else:
            self.threshold = threshold

        if max_threshold is None:
            self.max_threshold = self.num_predecessor * (self.dim_input/2)**2
        else:
            self.max_threshold = max_threshold
        self.save_alarms = False
        self.alarm_file = None
        self.alarm_filename = "Alarms.txt"
        self.change_rate = 0.0

    def create_odbodwindows(self):
        super(ODBODTHRESHOLD, self).create_odbodwindows()
        self.odbod_window_alarm = ODBODWindow(self.stat_type, y_tick_pos_max=1, y_tick_labels_interval=0.5)
        self.odbod_window_alarm.set_info_texts('Time vs Rate', 'Time', 'Rate')
        self.odbod_window_alarm.create()

    def process(self, x):
        super(ODBODTHRESHOLD, self).process(x)

        self.change_rate = float(self.current_distances_sum) / self.max_threshold
        # self.change_rate = min(1.0, self.change_rate)

        if self.__class__.__name__.lower() == 'odbodthreshold':
            if self.change_rate >= self.threshold:
                self.raise_alarm()
                return True
        return False

    def update_windows(self):
        super(ODBODTHRESHOLD, self).update_windows()
        self.odbod_window_alarm.update(self.change_rate, loss_msg='Change rate:', print_verbose=self.print_verbose)

    def set_alarm_save(self, filename=None):
        self.save_alarms = True
        if filename:
            self.alarm_filename = filename
        self.alarm_file = open(self.alarm_filename,"w")

    def raise_alarm(self):
        print("!!!!Alarm is activated at: ", self.t, "!!!!")
        if self.save_alarms:
            self.alarm_file.write(str(self.t) + "\n")

    def __del__(self):
        super(ODBODTHRESHOLD,self).__del__()
        if self.save_alarms:
            self.alarm_file.close()


class ODBODTHRESHOLDCHI2DISTANCE(ODBODTHRESHOLD):
    def __init__(self, stat_type=None, dim_input=1, num_predecessor=5, threshold=None, lambda_p=1.0, beta_p=1.0,
                 initial_mahalanobis_matrix=None, alpha_p=0.05):
        super(ODBODTHRESHOLDCHI2DISTANCE, self).__init__(stat_type, dim_input, num_predecessor, threshold, lambda_p,
                                                         beta_p, initial_mahalanobis_matrix)
        self.print_verbosely(False)
        if threshold is None:
            self.threshold = 1.0 - alpha_p
        else:
            self.threshold = threshold

        self.alpha = alpha_p
        self.degree_of_freedom = 0.0
        self.threshold_dist = 0.0
        self.degree_of_freedom = self.dim_input * self.num_predecessor
        self.threshold_dist = chi2.isf(self.alpha, self.degree_of_freedom)
        if self.print_verbose:
            print("Max threshold distance: ", self.threshold_dist)

    def process(self, x):
        super(ODBODTHRESHOLDCHI2DISTANCE, self).process(x)
        if self.data.shape[1] <= self.num_predecessor:
            self.degree_of_freedom = self.dim_input * self.data.shape[1]
            self.threshold_dist = chi2.isf(self.alpha, self.degree_of_freedom)
        else:
            self.degree_of_freedom = self.dim_input * self.num_predecessor
            self.threshold_dist = chi2.isf(self.alpha, self.degree_of_freedom)
        if self.print_verbose:
            print("Max threshold distance: ", self.threshold_dist)

        self.change_rate = 1.0 - chi2.sf(self.current_distances_sum, self.degree_of_freedom)
        if self.current_distances_sum >= self.threshold_dist:
            self.raise_alarm()

        if self.change_rate >= self.threshold:
            return True
        return False
