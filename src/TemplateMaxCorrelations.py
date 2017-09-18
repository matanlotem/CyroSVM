from TemplateUtil import shape_to_slices, get_normalize_template_dm
from Configuration import CONFIG
from CommonDataTypes import SixPosition

from scipy import signal
import numpy as np

class TemplateMaxCorrelations:
    """
    correlation_values is a list of length len(templates) each element is a Tomogram sized matrix containing the max correlation for each point to this template (over all tilts)
    best_tilts is a list of length len(templates) each element is a Tomogram sized matrix containing the tilt_id that resulted with the max correlation
    """

    def __init__(self, tomogram, templates):
        self.correlation_values = []
        self.best_tilt_ids = []
        for template_group in templates:
            max_correlation = np.zeros(tomogram.density_map.shape)
            max_correlation_tilt_id = np.zeros(max_correlation.shape, dtype = int)
            for tilted_template in template_group:
                correlation = self.create_fit_score(tilted_template, tomogram)
                max_correlation = np.maximum(correlation, max_correlation)
                max_correlation_tilt_id[max_correlation == correlation] = tilted_template.tilt_id
            self.correlation_values.append(max_correlation)
            self.best_tilt_ids.append(max_correlation_tilt_id)


    def suggest_best_3pos_and_tilt_in_neighborhood(self, candidate):
        """
        Searching in a constant neighborhood around the candidate's 3position,
        Only in the Correlation map of the candidate's label (this is the template_id)
        find the best correlation, and return the corresponding 3pos and tilt
        """
        if (candidate.label == None):
            raise Exception("Can't suggest best 3pos for unlabeled candidate", str(candidate))
        elif candidate.label >= len(self.correlation_values):
            raise Exception("Candidate's label out of bounds in TemplateMaxCorrelations:", str(candidate))
        else:
            dim = 2 if (self.correlation_values[0].shape[2] == 1) else 3
            if dim == 2:
                neighborhood_half_side = tuple([CONFIG.NEIGHBORHOOD_HALF_SIDE,CONFIG.NEIGHBORHOOD_HALF_SIDE,0])
            else:
                neighborhood_half_side = tuple([CONFIG.NEIGHBORHOOD_HALF_SIDE,CONFIG.NEIGHBORHOOD_HALF_SIDE,CONFIG.NEIGHBORHOOD_HALF_SIDE])

            cor_for_label = self.correlation_values[candidate.label]
            corner = [x[0] - x[1] for x in zip(candidate.get_position(),neighborhood_half_side)]
            relative_shape = [2*x+1 for x in neighborhood_half_side]
            shape = shape_to_slices(relative_shape, corner)
            if ([shape[i].start < 0 or shape[i].stop > cor_for_label.shape[i] for i in
                 range(len(cor_for_label.shape))].count(True) > 0):
                raise Exception("problem with shapes- NBRHD:", neighborhood_half_side, " candidate_pos:", candidate.get_position(), " returned shape:", shape)
            relative_flat_idx = np.argmax(cor_for_label[shape])
            relative_coordinates = np.unravel_index(relative_flat_idx, relative_shape)
            best_3pos = tuple([x[0] + x[1] for x in zip(corner,relative_coordinates)])
            corresponding_tilt = self.best_tilt_ids[candidate.label][best_3pos]
            return SixPosition(best_3pos, corresponding_tilt)

    def create_fit_score(self, raw_template, tomogram):
        normalized_dm = get_normalize_template_dm(raw_template)
        return signal.correlate(tomogram.density_map, normalized_dm, mode='same', method='fft')



