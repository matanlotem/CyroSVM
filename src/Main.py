import sys
import argparse
import pickle

from CommonDataTypes import EulerAngle
from TemplateGenerator import TemplateGenerator, TemplatesType
from TomogramGenerator import TomogramGenerator
from TemplateFactory import TemplateFactory
from TomogramFactory import TomogramFactory
from Configuration import CONFIG
from TomogramGenerator import generate_tomogram_with_given_candidates
from SvmTrain import svm_train
from SvmEval import svm_eval
from SvmView import svm_view


SUPPORTED_COMMANDS = ('generate', 'train', 'eval', 'view_results')
GENERETABLES = ('templates', 'tomograms')


# Generate wrapper functions
def _generate_templates(generator, out_path, angular_resolution, template_type):
    tf = TemplateFactory(generator)
    tf.set_paths(out_path)
    tf.set_angular_resolution(angular_resolution)
    tf.set_template_type(template_type)
    templates = tf.build()

    if not generator == TemplateGenerator.CHIMERA:
        save_generator = TemplateFactory(TemplateGenerator.LOAD).set_paths([out_path]).set_save(True).build()
        for template, save in zip(templates, save_generator):
            save(template)


def _generate_tomograms(generator, template_paths, out_path, criteria, num_tomograms):
    # Load templates
    templates = TemplateFactory(TemplateGenerator.LOAD_3D).set_paths(template_paths).build()

    # Generate tomograms
    tf = TomogramFactory(generator).set_templates(templates).set_paths(out_path).set_criteria(criteria)
    tf.set_num_tomograms(num_tomograms)
    tomograms = tf.build()

    # Save tomograms
    # save_generator = TomogramFactory(TomogramGenerator.LOAD).set_paths(out_path).set_save(True).build()
    # for tomogram, save in zip(tomograms, save_generator):
    #     save(tomogram)
    with open(out_path[0], 'wb') as file:
        pickle.dump(list(tomograms), file)


# Train wrapper function
def _train(svm_path, template_paths, tomogram_paths):
    # Load the templates and the tomograms
    templates_generator = TemplateFactory(TemplateGenerator.LOAD_3D).set_paths(template_paths).build()
    tf = TomogramFactory(TomogramGenerator.LOAD).set_paths(tomogram_paths)
    tomograms_generator = tf.build()

    # Create and train the SVM
    svm = svm_train(templates_generator, tomograms_generator)

    # Save the SVM
    with open(svm_path, 'wb') as file:
        pickle.dump(svm, file)


# Eval wrapper function
def _eval(svm_path, tomogram_paths, out_path):
    # Load the SVM
    with open(svm_path, 'rb') as file:
        svm_and_templates = pickle.load(file)

    # Load the tomograms
    tf = TomogramFactory(TomogramGenerator.LOAD)
    tf.set_paths(tomogram_paths)
    tomograms = list(tf.build())

    # Evaluate the tomograms
    eval_result = svm_eval(svm_and_templates, tomograms)

    # Save results
    with open(out_path, 'wb') as file:
        pickle.dump((svm_and_templates[1][0], tomograms, svm_and_templates[1][1], eval_result), file)


# View results wrapper function
def _view_results(result_path):
    # Load results
    with open(result_path, 'rb') as file:
        result = pickle.load(file)
    EulerAngle.Tilts, evaluation_tomograms, templates, output_candidates = result

    # Show results
    svm_view(evaluation_tomograms, templates, output_candidates)


def main(argv):
    parser = argparse.ArgumentParser(description='Train or evaluate an SVM to classify electron density maps.')
    subparsers = parser.add_subparsers(dest='command', help='Command to initiate.')

    # TODO: fine-tune the generate commands to support all generation methods. That's including accepting required parameters
    # Define generate sub-command
    generator_parser = subparsers.add_parser(SUPPORTED_COMMANDS[0])
    generator_subparser = generator_parser.add_subparsers(dest='generetable', help='Type of object to generate.')

    # Define generate templates sub-sub-command
    template_parser = generator_subparser.add_parser(GENERETABLES[0])
    template_parser.add_argument('generator', choices=TemplateGenerator.keys(),
                                 help='The kind of generator to use when creating the templates')
    template_parser.add_argument('out_path', nargs=1, help='Path to save in the generated template')
    template_parser.add_argument('-r', '--angular_resolution', type=int, help='Angular resolution')
    template_parser.add_argument('-t', '--template_type', choices=TemplatesType.keys(),
                                 help='Type of templates to be generated')

    # Define generate tomograms sub-sub-command
    tomogram_parser = generator_subparser.add_parser(GENERETABLES[1])
    tomogram_parser.add_argument('generator', choices=TomogramGenerator.keys(),
                                 help='The kind of generator to use when creating the tomograms')
    tomogram_parser.add_argument('template_paths', help='Path to the templates to be used in the generation')
    tomogram_parser.add_argument('out_path', nargs='+', help='Path to save in the generated tomograms')
    tomogram_parser.add_argument('-c', '--criteria', nargs='+', type=int,
                                 help='Criteria to be used by the tomogram generator')
    tomogram_parser.add_argument('-n', '--num_tomograms', type=int,
                                 help='Number of tomograms to create')

    # Define train sub-command
    train_parser = subparsers.add_parser(SUPPORTED_COMMANDS[1])
    train_parser.add_argument('svm_path', metavar='svm', type=str,
                              help='Path to save in the created svm.')
    train_parser.add_argument('-t', '--templatepath', metavar='templatepath', dest='template_paths',
                              type=str,
                              required=True,
                              help='Paths to the templates to be trained with.')
    train_parser.add_argument('-d', '--datapath', metavar='datapath', dest='tomogram_paths', type=str, required=True,
                              help='Paths to the tomograms to be trained on. If -g is supplied then datapath will be '
                                   'ignored.')

    # Define eval sub-command
    eval_parser = subparsers.add_parser(SUPPORTED_COMMANDS[2])
    eval_parser.add_argument('svm_path', metavar='svm', type=str,
                             help='Path to the pickle of the svm to use. (As returned by the train subcommand)')
    eval_parser.add_argument('-d', '--datapath', metavar='datapath', dest='tomogram_paths', type=str,
                             required=True,
                             help='Path to the tomograms to be evaluated.')
    eval_parser.add_argument('-o', '--outpath', dest='out_path', type=str, required=True,
                             help='Path to which the results will be saved.')

    # Define view results sub-command
    results_parser = subparsers.add_parser(SUPPORTED_COMMANDS[3])
    results_parser.add_argument('result_path', metavar='result', type=str,
                                help='Path to the pickle of the result to show. (As returned by the eval subcommand)')

    # Parse the arguments
    args = parser.parse_args(argv)

    # Run the appropriate command
    if args.command == SUPPORTED_COMMANDS[0]:
        if args.generetable == GENERETABLES[0]:
            _generate_templates(args.generator, args.out_path, args.angular_resolution, args.template_type)
        elif args.generetable == GENERETABLES[1]:
            _generate_tomograms(args.generator, args.template_paths, args.out_path, args.criteria, args.num_tomograms)
    elif args.command == SUPPORTED_COMMANDS[1]:
        _train(args.svm_path, args.template_paths, args.tomogram_paths)
    elif args.command == SUPPORTED_COMMANDS[2]:
        _eval(args.svm_path, args.tomogram_paths, args.out_path)
    elif args.command == SUPPORTED_COMMANDS[3]:
        _view_results(args.result_path)
    else:
        raise NotImplementedError('Command %s is not implemented.' % args.command)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        # main(['generate', 'templates', 'CHIMERA', 'C:\\dev\\Anaconda\\CryoEmSvm\\Chimera\\GeometricTemplates\\',
        #       '-r', '60', '-t', 'GEOMETRIC_3D'])
        # main(['generate', 'tomograms', 'RANDOM', 'C:\\dev\\Anaconda\\CryoEmSvm\\Chimera\\GeometricTemplates\\',
        #       'C:\\dev\\Anaconda\\CryoEmSvm\\Chimera\\GeometricTemplates\\tomograms.pkl', '-c', '2', '3', '-n', '1'])
        # print('Training')
        # main(['train', 'my_svm.pkl', '-t', 'C:\\dev\\Anaconda\\CryoEmSvm\\Chimera\\GeometricTemplates\\',
        #       '-d', 'C:\\dev\\Anaconda\\CryoEmSvm\\Chimera\\GeometricTemplates\\tomograms.pkl'])
        # main(['generate', 'tomograms', 'RANDOM', 'C:\\dev\\Anaconda\\CryoEmSvm\\Chimera\\GeometricTemplates\\',
        #       'C:\\dev\\Anaconda\\CryoEmSvm\\Chimera\\GeometricTemplates\\eval_tomograms.pkl', '-c', '3', '3', '-n', '1'])
        # print('Evaluating')
        # main(['eval', 'my_svm.pkl', '-d', 'C:\\dev\\Anaconda\\CryoEmSvm\\Chimera\\GeometricTemplates\\eval_tomograms.pkl',
        #       '-o', 'my_result.pkl'])
        # print('Show results')
        main(['view_results', 'my_result.pkl'])
        pass
    else:
        main(None)
