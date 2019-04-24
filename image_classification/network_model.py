import scripts.retrain
import tensorflow as tf
from argparse import Namespace
import numpy as np
import zipfile
from bo.models import NetworkParameter
from bo.network_dao import add_network_parameter

# from config.config import tf_files_path
tf_files_path = '.'


def train_model(image_path):
    print('training model started')
    custom_path = image_path.replace('.zip', '')
    custom_path = image_path[0:image_path.rfind('/')]
    zip_ref = zipfile.ZipFile(image_path, 'r')
    zip_ref.extractall(custom_path)
    zip_ref.close()

    print('Start model training')
    args = {'architecture': 'mobilenet_0.50_224', 'bottleneck_dir': tf_files_path + 'tf_files/bottlenecks',
            'eval_step_interval': 10, 'final_tensor_name': 'final_result', 'flip_left_right': False,
            'how_many_training_steps': 500, 'image_dir': image_path.replace('.zip', ''),
            'intermediate_output_graphs_dir': '/tmp/intermediate_graph/', 'intermediate_store_frequency': 0,
            'learning_rate': 0.01, 'model_dir': tf_files_path + 'tf_files/models/',
            'output_graph': tf_files_path + 'tf_files/retrained_graph.pb',
            'output_labels': tf_files_path + 'tf_files/retrained_labels.txt', 'print_misclassified_test_images': False,
            'random_brightness': 0, 'random_crop': 0, 'random_scale': 0,
            'summaries_dir': tf_files_path + 'tf_files/training_summaries/mobilenet_1.0_224', 'test_batch_size': -1,
            'testing_percentage': 10, 'train_batch_size': 100, 'validation_batch_size': 100,
            'validation_percentage': 10}

    print('User parameters:')
    print(args)
    flags = Namespace(**args)
    scripts.retrain.FLAGS = flags
    scripts.retrain.main(args)
    print('Model training finnished')


# ##################


def load_graph(model_file):
    print('Loading model.')
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def load_trained_model(network):
    train_data_path, is_trained = check_if_trained(network)

    if not is_trained:
        train_model(train_data_path)


    global global_graph
    global_graph = load_graph(tf_files_path + '/tf_files/retrained_graph.pb')
    print('Model loaded')

    global global_labels
    global_labels = load_labels(
        tf_files_path + '/tf_files/retrained_labels.txt')  # except FileNotFoundError:  #     print('Model or labels file not found')

def check_if_trained(network):
    trained = False
    train_path = ""

    for parameter in network.parameters:
        if parameter.abbreviation == "IS_TRAINED" and parameter.value is not None and parameter.value.lower() == 'true':
            trained = True

        if parameter.abbreviation == "TRAIN_DATA_PATH":
            train_path = parameter.value

    if trained is False:
        train_model(train_path)
        parameter = NetworkParameter('IS_TRAINED', 'IS_TRAINED', True, network.network_id)
        add_network_parameter(parameter)
    return train_path, trained


def prepare_image(image, target, input_mean=0, input_std=255):
    float_caster = tf.cast(image, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [target[0], target[1]])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


def predict(image, additional_info):
    image = prepare_image(image, target=(224, 224), input_mean=128, input_std=128)

    with tf.Session(graph=global_graph) as sess:
        input_layer = "input"
        output_layer = "final_result"
        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        input_operation = global_graph.get_operation_by_name(input_name)
        output_operation = global_graph.get_operation_by_name(output_name)

        results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: image})

    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]

    data = {"predictions": []}
    for i in top_k:
        r = {"label": global_labels[i], "probability": float(results[i])}
        data["predictions"].append(r)

    return data
