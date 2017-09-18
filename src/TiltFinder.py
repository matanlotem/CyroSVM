from Labeler import JUNK_ID


class TiltFinder:
    """
    This class matches a candidate with the best tilt that corresponds to its label
    """
    def __init__(self, max_correlations):
        """
        :param max_correlations: a data structure of type TemplateMaxCorrelations initialized according to the relevant tomogram and templates
        """
        self.max_correlations = max_correlations

    def find_best_tilt(self, candidate):
        if candidate.label == JUNK_ID:
            return #what should we return in case of junk? does it matter?
        candidate.six_position.tilt_id = self.max_correlations.best_tilt_ids[candidate.label][candidate.six_position.COM_position]


if __name__ == '__main__':
    pass