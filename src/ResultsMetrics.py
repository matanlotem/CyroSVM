from Labeler import PositionLabeler, JUNK_ID
from Configuration import CONFIG
from CommonDataTypes import EulerAngle

import numpy as np
import math


class ResultsMetrics:
    def __init__(self, ground_truth_composition, evaluated_composition):
        self.ground_truth_composition = ground_truth_composition
        self.evaluated_composition = evaluated_composition
        self.evaluation_results = {'true_label_and_tilt':[],
                                   'true_label_false_tilt':[],
                                   'true_junk':[],
                                   'wrong_label':[],
                                   'false_junk':[],
                                   'false_existenece': [],
                                   'missing_ground_truth': []}
        self.get_evalutaion_results()

    def get_evalutaion_results(self):
        labeler = PositionLabeler(self.ground_truth_composition)
        ground_truth_matches = []
        for candidate in self.evaluated_composition:
            labeler.label(candidate, set_label = False)
            if candidate.label != JUNK_ID: # not identified as junk
                if candidate.ground_truth_match is not None and \
                                candidate.ground_truth_match.label == candidate.label: # true match
                    if self.tilts_match(candidate.ground_truth_match, candidate): # true tilt
                        self.evaluation_results['true_label_and_tilt'].append(candidate)
                    else: # false tilt
                        self.evaluation_results['true_label_false_tilt'].append(candidate)
                elif candidate.ground_truth_match is not None: # has matching ground truth but wrong label
                    self.evaluation_results['wrong_label'].append(candidate)
                else: # does not have matching ground truth
                    self.evaluation_results['false_existenece'].append(candidate)
            else: # idenetified as junk
                if candidate.ground_truth_match is None: # true junk (has no matching ground truth)
                    self.evaluation_results['true_junk'].append(candidate)
                else: # false junk (has matching ground truth)
                    self.evaluation_results['false_junk'].append(candidate)

            # all ground truth candidates that have a match
            if candidate.ground_truth_match is not None:
                ground_truth_matches.append(candidate.ground_truth_match)

        # ground truths with no matched candidate
        for candidate in self.ground_truth_composition:
            if candidate not in ground_truth_matches:
                self.evaluation_results['missing_ground_truth'].append(candidate)

        # ground truths with multiple matched candidate
        #Todo


    def gt_position_dist(self, candidate):
        return self.position_dist(candidate,candidate.ground_truth_match)


    def gt_tilt_dist(self, candidate):
        return self.tilt_dist(candidate, candidate.ground_truth_match)


    def tilts_match(self, c1, c2):
        (rel_angle_3d, rel_psi) = self.tilt_dist(c1,c2)
        return (abs(rel_angle_3d) < CONFIG.TILT_THRESHOLD and abs(rel_psi) < CONFIG.TILT_THRESHOLD)


    def position_dist(self, c1, c2):
        if c2 is None:
            return 0
        return (np.sum((np.array(c1.get_position()) - np.array(c2.get_position()))**2))**0.5


    def tilt_dist(self, c1, c2):
        if c2 is None:
            return (0,0)
        a1 = EulerAngle.Tilts[c1.get_tilt_id()]
        a2 = EulerAngle.Tilts[c2.get_tilt_id()]
        T1 = math.radians(a1.Theta)
        T2 = math.radians(a2.Theta)
        P1 = math.radians(a1.Phi)
        P2 = math.radians(a2.Phi)
        rel_angle_3d = math.acos(math.sin(T1)*math.sin(T2) * (math.cos(P1)*math.cos(P2) + math.sin(P1)*math.sin(P2)) + math.cos(T1)*math.cos(T2))

        rel_psi = a2.Psi-a1.Psi
        if abs(rel_angle_3d) < 1e-7: # for 2d
            rel_psi =+ a2.Phi-a1.Phi

        return (math.degrees(rel_angle_3d), rel_psi)


    def print_summary(self):
        for key, val in self.evaluation_results.items():
            print(key,':\t',len(val))


    def print_full_stats(self, short=True):
        for key, val in self.evaluation_results.items():
            if len(val) > 0:
                print(key,'\n===============')
                for candidate in val:

                    if short:
                        self.print_candidate_stats_short(candidate)
                    else:
                        self.print_candidate_stats(candidate)
                print('')

        self.print_summary()


    def print_candidate_stats(self, candidate):
        if candidate.ground_truth_match:
            dist = self.gt_position_dist(candidate)
            tilt_dist = self.gt_tilt_dist(candidate)
            print('Template ', candidate.label, '(gt', candidate.ground_truth_match.label,')')
            print('\tDistance: ',dist,  '\tPos ', candidate.get_position(), '(gt', candidate.ground_truth_match.get_position(),')')
            print('\tTilt Distance: ', tilt_dist, '\tTilt ', candidate.get_euler_angle(), '(gt', candidate.ground_truth_match.get_euler_angle(),')')

    def print_candidate_stats_short(self, candidate):
        if candidate.ground_truth_match:
            dist = self.gt_position_dist(candidate)
            tilt_dist = self.gt_tilt_dist(candidate)
            print('Label: ',candidate.label, candidate.ground_truth_match.label, '\tDist: ', dist,
                  '\tPos: ', candidate.get_position() ,'\tTiltDist: ', tilt_dist)