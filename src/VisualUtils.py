from Labeler import JUNK_ID


import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


"""
This file contains utility methods from visualization
"""

# ----------------------- 2d -----------------------------
def print_candidate_list(candidates):
    print("There are " + str(len(candidates)) + " candidates")
    #print true and false seperately? using position labeler?
    print([c.six_position.COM_position for c in candidates])


def candidates2dm(candidates, shape):
    peaks = np.zeros(shape)
    for c in candidates:
        peaks[c.six_position.COM_position] = 1
    return peaks


def show_densitymap(dm, title, subplot=111):
    ax = plt.subplot(subplot)
    fig = plt.gcf()
    fig.suptitle(title)
    if len(dm.shape) == 3:
        ax.imshow(dm[:, :, 0])
    else:
        ax.imshow(dm)

def show_templates(templates):
    # TODO: Broken for 4 templates and over (subplot number)
    print("There are " + str(len(templates)) + " templates-")
    fig = plt.figure(1)
    fig.suptitle("Templates")
    SHOW_N_TILTS = 3
    for i in range(len(templates)):
        for j in range(SHOW_N_TILTS):
            #this fits them to len(template) rows each containing SHOW_N_TILTS plots
            ax = plt.subplot(int(str(len(templates)) + str(SHOW_N_TILTS) + str(i * SHOW_N_TILTS + j + 1)))
            ax.imshow(templates[i][j].density_map[:, :, 0])
    plt.show()

def show_tomogram(tomogram, criteria):
    print('This is the generated tomogram for criteria: ' + str(criteria))
    print('The tomogram composition is: ' + str(tomogram.composition))
    fig = plt.figure(2)
    fig.suptitle("Tomogram")
    ax = plt.subplot()
    ax.imshow(tomogram.density_map[:, :, 0])
    plt.show()

def show_candidates(selector, candidates, tomogram):
    print_candidate_list(candidates)

    fig = plt.figure(3)
    fig.suptitle("Candidate selection")

    ax = plt.subplot(221)
    ax.set_title('Original Tomogram')
    ax.imshow(tomogram.density_map[:, :, 0])

    ax = plt.subplot(222)
    ax.set_title('Max Correlation')
    ax.imshow(selector.max_correlation_per_3loc[:,:,0])

    ax = plt.subplot(223)
    ax.set_title('Blurred Correlation')
    ax.imshow(selector.blurred_correlation_array[:,:,0])

    ax = plt.subplot(224)
    ax.set_title('Selected Candidates')
    dm = candidates2dm(candidates, tomogram.density_map.shape)
    ax.imshow(dm[:, :, 0])

    plt.show()

def compare_reconstruced_tomogram(truth_tomogram, recon_tomogram, plot_dif_map = False):
    fig = plt.figure(2)
    ax = plt.subplot(121)
    ax.set_title('Truth Tomogram')
    ax.imshow(truth_tomogram.density_map[:, :, 0])

    ax = plt.subplot(122)
    ax.set_title('Reconstructed Tomogram')
    ax.imshow(recon_tomogram.density_map[:, :, 0])
    plt.show()

    if not plot_dif_map:
        return
    fig = plt.figure(3)
    ax = plt.subplot()
    ax.set_title('Overlap')
    ax.imshow(truth_tomogram.density_map[:, :, 0] - recon_tomogram.density_map[:, :, 0])
    plt.show()

def compare_candidate_COM(truth_candidates, reco_candidates, tomogram):
    map = tomogram.density_map
    for c in truth_candidates:
        pos = c.six_position.COM_position
        map[pos[0], pos[1], 0] += 2
    for c in reco_candidates:
        pos = c.six_position.COM_position
        if c.label == JUNK_ID:
            map[pos[0], pos[1], 0] -= 1
        else:
            map[pos[0], pos[1], 0] += 1
    fig = plt.figure(2)
    fig.suptitle("Centers")
    ax = plt.subplot()
    ax.imshow(map[:, :, 0])
    plt.show()


def analyzer_max_correltations(analyzer):
    maxmax = analyzer.max_correlations.correlation_values[0]
    for cv in analyzer.max_correlations.correlation_values:
        maxmax = np.maximum(maxmax, cv)
    show_densitymap(maxmax, 'max correlations')


# ----------------------- 3D -----------------------------
def show3d(dm):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    d = int(dm.shape[2]**0.5)+1
    fig, axarr = plt.subplots(d,d)
    for z in range(dm.shape[2]):
        zdm = np.copy(dm[:,:,z])
        zdm[0,0] = 1
        axarr[z//d, z%d].imshow(zdm)
    plt.show()

def slider3d(dm):
    ax = plt.subplot()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    plt.subplots_adjust()
    zdm = np.copy(dm[:, :, 0])
    zdm[0, 0] = 1
    l = plt.imshow(zdm, vmax=dm.max(), vmin=dm.min())

    axframe = plt.axes([0.25, 0.1, 0.65, 0.03])
    sframe = Slider(axframe, 'Frame', 0, dm.shape[0]-1, valinit=0)

    def update(val):
        zdm = np.copy(dm[:, :, int(np.around(sframe.val))])
        zdm[0, 0] = 1
        l.set_data(zdm)

    sframe.on_changed(update)
    plt.show()


def show_candidates3D(selector, candidates, tomogram):
    print_candidate_list(candidates)

    slider3d(tomogram.density_map)

    slider3d(selector.max_correlation_per_3loc)

    slider3d(selector.blurred_correlation_array)

    dm = candidates2dm3D(candidates, tomogram.density_map.shape)
    slider3d(dm)

def candidates2dm3D(candidates, shape):
    peaks = np.zeros(shape)
    for c in candidates:
        for i in range(-2,2):
            for j in range(-2,2):
                for k in range(-2,2):
                    peaks[c.six_position.COM_position] += 1
    return peaks