# This is an example usage for the python flow
# We first make sure that there is a file called config.txt in the examples directory.

from TemplateGenerator import load_templates_3d, generate_templates_3d, TemplatesType
from TomogramGenerator import generate_random_tomogram_set, generate_random_tomogram
from SvmTrain import svm_train
from SvmEval import svm_eval
from SvmView import svm_view

# Say that we already created templates in Chimera/GeometricTemplates
# otherwise use:
#   templates = generate_templates_3d('../Chimera/GeometricTemplates/', 60, TemplatesType.GEOMETRIC_3D)
# !!! remember to set chimeraUtils path in config file
templates = load_templates_3d('../Chimera/GeometricTemplates/')

# We generate 2 training tomograms and 1 evaluation tomogram. We will use different criteria for the different tomograms
train_tomograms = generate_random_tomogram_set(templates, [3, 3, 3], 2, 3)
eval_tomogram = [generate_random_tomogram(templates, [1, 4, 2], 3)]

# We will train an SVM on the tempaltes and tomograms
svm = svm_train(templates, train_tomograms)

# After we have trained an SVM we evaluate the tomogram
candidates = svm_eval(svm, eval_tomogram)

# Then we can view the results
svm_view(eval_tomogram, templates, candidates)
