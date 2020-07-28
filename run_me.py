import numpy as np

from distance_changepoint_detectors import ODBODTHRESHOLD, ODBODTHRESHOLDCHI2DISTANCE, CustomODBODHistogram


_author__ = 'murat.semerci@boun.edu.tr'


class ChangeDistanceDetectorThreshold:
    def __init__(self, dim_input=10, num_predecessor=9, lambda_p=1.0, beta_p=4.0, threshold=None):
        self.dim = dim_input  # input dimension
        self.num_predecessor = num_predecessor  # number of predecessor
        self.odbod = ODBODTHRESHOLD(['PacketHistogram'], self.dim, self.num_predecessor,
                                    threshold, lambda_p, beta_p, initial_mahalanobis_matrix=None,
                                    max_threshold=40.0)
        result_file = "distances.txt"
        alarm_file = "alarms.txt"
        self.odbod.print_verbosely(True)
        self.odbod.set_result_save(result_file)
        self.odbod.set_alarm_save(alarm_file)
        self.odbod.create_odbodwindows()

    def handle_data_vector(self, data_vector):
        self.odbod.process(data_vector)
        self.odbod.update_windows()


class ChangeDistanceDetectorChiSquared:
    def __init__(self, dim_input=10, num_predecessor=9, lambda_p=1.0, beta_p=4.0, threshold=None):
        self.dim = dim_input  # input dimension
        self.num_predecessor = num_predecessor  # number of predecessor
        self.odbod = ODBODTHRESHOLDCHI2DISTANCE(['PacketHistogram'], dim_input=self.dim,
                                                num_predecessor=self.num_predecessor,
                                                threshold=1, lambda_p=lambda_p, beta_p=beta_p)
        result_file = "distances.txt"
        alarm_file = "alarms.txt"
        self.odbod.print_verbosely(True)
        self.odbod.set_result_save(result_file)
        self.odbod.set_alarm_save(alarm_file)
        self.odbod.create_odbodwindows()

    def handle_data_vector(self, data_vector):
        self.odbod.process(data_vector)
        self.odbod.update_windows()


def main():
    num_of_features = 3
    num_of_samples = 100
    max_counts = 100
    feature_names = ['Feat1', 'Feat2', 'Feat3']
    cpd = ChangeDistanceDetectorChiSquared(dim_input=num_of_features)
    # cpd = ChangeDistanceDetectorThreshold(dim_input=num_of_features)
    cstWindows = CustomODBODHistogram(headers=feature_names)
    cstWindows.set_max_value(max_value=max_counts)
    for _ in range(num_of_samples):
        values = np.random.randint(0, max_counts, size=(num_of_features))
        cpd.handle_data_vector(values)
        cstWindows.update(values)


if __name__ == '__main__':
    main()
