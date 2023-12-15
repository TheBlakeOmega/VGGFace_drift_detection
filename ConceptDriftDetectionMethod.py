from river.drift import PageHinkley, ADWIN
import numpy as np
import matplotlib.pyplot as plt


class ConceptDriftDetectionMethod:

    def __init__(self, method):
        self.monitored_variable_file = open("results_50_people/" + method + '_F1Scores.txt', "w")
        self.drifts_file = open("results_50_people/" + method + '_Drifts.txt', "w")
        self.detection_method = None
        self.method_name = method
        if method == "page-hinkley":
            self.detection_method = PageHinkley(min_instances=20, threshold=5)
        if method == "adwin":
            self.detection_method = ADWIN(clock=1)

    def test(self, new_value):
        self.monitored_variable_file.write(str(new_value) + " ")
        self.detection_method.update(new_value)
        if self.detection_method.drift_detected:
            self.drifts_file.write(str(new_value) + " ")
            return True
        self.drifts_file.write(str(-1) + " ")
        return False

    def updatePHT(self, values):
        self.detection_method = PageHinkley(min_instances=20, threshold=25)
        for val in values:
            self.detection_method.update(val)

    def closeTest(self):
        self.monitored_variable_file.close()
        self.drifts_file.close()

    def makePlot(self):
        with open("results_50_people/" + self.method_name + '_F1Scores.txt', "r") as file:
            string = file.read()
        errors = np.fromstring(string, dtype=float, sep=" ")
        with open("results_50_people/" + self.method_name + '_Drifts.txt', "r") as file:
            string = file.read()
        drifts = np.fromstring(string, dtype=float, sep=" ")
        plt.title("Evaluation f-1 score of VGGModel")
        xs = np.array(list(range(1, len(errors) + 1)))
        plt.plot(xs, errors, label="Loss function values")
        plt.plot(xs[drifts > -1], drifts[drifts > -1], marker='D', linestyle='None',
                 label=self.method_name + " test change point")
        plt.legend(loc='best')
        plt.savefig("Model's performance drifts with " + self.method_name + ".png")
        plt.close()
