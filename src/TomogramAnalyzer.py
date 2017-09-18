from TemplateMaxCorrelations import TemplateMaxCorrelations
from CandidateSelector import CandidateSelector
from FeaturesExtractor import FeaturesExtractor
from TiltFinder import TiltFinder

class TomogramAnalyzer:
    """
    This class encapsulates the entire tomogram analyses process
     Candidate selection
     feature extraction
     labeling
     improvement using neighborhood
    """
    def __init__(self, tomogram, templates, labeler):
        self.tomogram = tomogram
        self.templates = templates
        self.labeler = labeler
        self.max_correlations = None

    def analyze(self):
        print('\tcaclualting correlations')
        self.max_correlations = TemplateMaxCorrelations(self.tomogram, self.templates)

        candidate_selector = CandidateSelector(self.max_correlations, self.templates[0][0].density_map.shape)
        features_extractor = FeaturesExtractor(self.max_correlations)
        tilt_finder = TiltFinder(self.max_correlations)

        print('\tselecting candidates')
        candidates = candidate_selector.select(self.tomogram)
        feature_vectors = []
        labels = []

        print('\tlabeling and extracting features')
        for candidate in candidates:
            feature_vectors.append(features_extractor.extract_features(candidate))
            # this sets each candidate's label
            labels.append(self.labeler.label(candidate))
            candidate.six_position = self.max_correlations.suggest_best_3pos_and_tilt_in_neighborhood(candidate)

        return candidates, feature_vectors, labels