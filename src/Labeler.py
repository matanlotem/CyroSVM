from Configuration import CONFIG

import numpy as np

# The label that corresponds to NOTHING_IS_HERE
JUNK_ID = -1

class Labeler:
    """
    An entity that matches a candidate with a label (either using an SVM or a known set of GroundTruth candidates)
    """

    def label(self, candidate):

        return None

class PositionLabeler:
    """
        An entity that matches a candidate with a label using a known set of GroundTruth candidates
    """
    def __init__(self, composition):
        self.threshold = CONFIG.DISTANCE_THRESHOLD
        self.composition = composition
        #elements of composition who were found during labeling
        self.associated_composition = []



    def label(self, candidate, set_label = True):
        """
           label according to proximity to Ground truth
        """
        candidate_label = JUNK_ID
        ground_truth_candidate = None
        reals = []
        for real_candidate in self.composition:
            dist_squared = np.linalg.norm(np.array(candidate.six_position.COM_position) - np.array(real_candidate.six_position.COM_position))
            if dist_squared <= self.threshold:
                reals.append((dist_squared, real_candidate))
        if (len(reals) > 0):
            ground_truth_candidate = min(reals,key=lambda x: x[0])[1]

        if ground_truth_candidate:
            candidate_label = ground_truth_candidate.label

        if set_label:
            candidate.set_label(candidate_label)
        candidate.set_ground_truth(ground_truth_candidate)

        return candidate_label


class SvmLabeler(Labeler):
    """
    use an already trained SVM to predict the label
    """
    def __init__(self, svm):
        self.svm = svm

    def label(self, candidate):
        candidate_label = self.svm.predict(np.array([candidate.features]))
        candidate.set_label(candidate_label[0]) #for some reason candidate label is an array
        return candidate_label
