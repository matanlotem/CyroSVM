import MetricTester

class Metrics:
    def __init__(self):
        self.n_label_match  = 0
        self.n_mislabled    = 0
        self.n_svm_fn       = 0         #svm false negatives
        self.n_cd_fn        = 0         #candidate detector false negatives
        self.n_fp           = 0         #detected non junk candidate where none existed
        self.n_tilt_match   = 0
        self.n_true_junk    = 0
        self.n_tomograms    = 0

    def init_from_tester(self, metric_tester):
        self.n_label_match  = len(metric_tester.true_label)          #template detected and correctly labeled
        self.n_mislabled    = len(metric_tester.wrong_label)         #how many candidates were detected, but mislabeled
        self.n_svm_fn       = len(metric_tester.false_junk)          #svm false negatives
        self.n_cd_fn        = len(metric_tester.not_detected)        #candidate detector false negatives
        self.n_fp           = len(metric_tester.false_existence)     #detected non junk candidate where none existed
        self.n_tilt_match   = len(metric_tester.get_tilt_candidate_list(get_tolerated=True))
        self.n_true_junk    = len(metric_tester.success_junk)
        self.n_tomograms    = 1

    def merge(self, other_metric):
        self.n_label_match  += other_metric.n_label_match
        self.n_mislabled    += other_metric.n_mislabled
        self.n_svm_fn       += other_metric.n_svm_fn
        self.n_cd_fn        += other_metric.n_cd_fn
        self.n_fp           += other_metric.n_fp
        self.n_tilt_match   += other_metric.n_tilt_match
        self.n_true_junk    += other_metric.n_true_junk
        self.n_tomograms    += other_metric.n_tomograms

    def print_stat(self):
        n_total = self.n_label_match + self.n_mislabled + self.n_svm_fn + self.n_cd_fn #number of reconstucted templates
        svm_success_rate = self.n_label_match*100.0/n_total
        print("total number of tomograms tested: {}".format(self.n_tomograms))
        print("total number of templates reconstucted: {}".format(n_total))
        print("svm success rate = {:.2f}%".format(svm_success_rate))
        print("mislabeled templates = {} ({:.2f}%)".format(self.n_mislabled, self.n_mislabled*100.0/n_total))
        print("svm false negative = {} ({:.2f}%)".format(self.n_svm_fn, self.n_svm_fn*100.0/n_total))
        print("undetected candidates = {} ({:.2f}%)".format(self.n_cd_fn, self.n_cd_fn*100.0/n_total))
        print("number of false positives = {}".format(self.n_fp))
        print("number of true junk = {}".format(self.n_true_junk))
        if self.n_true_junk + self.n_fp != 0:
            print("true junk detection rate = {:.2f}%".format(self.n_true_junk*100.0/(self.n_true_junk + self.n_fp)))
        if self.n_label_match != 0:
            print("tilt match rate = {0:.2f}%".format(self.n_tilt_match*100.0/self.n_label_match))
        