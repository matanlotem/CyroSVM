from Labeler import SvmLabeler
from TomogramAnalyzer import TomogramAnalyzer
from CommonDataTypes import EulerAngle

def svm_eval(svm_and_templates, tomograms, test_list = None):
    """
    Evaluate the tomograms the supplied SVM and templates (As returned from svm_train).
    :param svm_and_templates:   SVM and templates as returned from svm_train.
    :param tomograms:           Iterator of the tomograms to be evaluated.
    :param test_list:           If not None, fills with analyzer objects for debugging
    :return A list of lists of the candidates for each tomogram.
    """

    svm, tilts_templates = svm_and_templates
    EulerAngle.Tilts, templates = tilts_templates
    tomogram_candidates = []
    labeler = SvmLabeler(svm)

    for tomogram in tomograms:
        # Analyze the tomogram
        analyzer = TomogramAnalyzer(tomogram, templates, labeler)
        (candidates, feature_vectors, predicted_labels) = analyzer.analyze()

        # Add the candidates to the list of results
        tomogram_candidates.append(candidates)

        # save analyzer object for debugging
        if test_list is not None:
            test_list.append(analyzer)

    return tomogram_candidates
