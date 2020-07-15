import os
import gdown
import zipfile
import tensorflow
from keras.models import Model, model_from_json
from keras.layers import Activation, Dense, Input, \
    Conv2DTranspose, Dense, Flatten
from keras.optimizers import Adam
import numpy as np
import cv2
import gradio

try:
    from keras_contrib.layers.normalization import \
        InstanceNormalization
except Exception:
    from keras_contrib.layers.normalization.instancenormalization\
        import \
        InstanceNormalization

CHECKPOINT_PATH = 'checkpoint/'
CHECKPOINT_LINK = 'https://drive.google.com/u/0/uc?' \
                  'export=download&confirm=a7YF&id=' \
                  '1MfXsRwjx5CTRGBoLx154S0h-Q3rIUNH0'

INPUT_SHAPE = (256, 256, 3)
# 25% i.e 64 width size will be mask from both side
MASK_PERCENTAGE = .25

g_input_shape = (INPUT_SHAPE[0], int(INPUT_SHAPE[1] * (MASK_PERCENTAGE * 2)),
                 INPUT_SHAPE[2])


EPSILON = 1e-9
ALPHA = 0.0004

def dcrm_loss(y_true, y_pred):
    return -tensorflow.reduce_mean(tensorflow.log(tensorflow.maximum(y_true, EPSILON)) +
                           tensorflow.log(tensorflow.maximum(1. - y_pred, EPSILON)))

d_input_shape = (INPUT_SHAPE[0], int(INPUT_SHAPE[1] * (MASK_PERCENTAGE *2)), INPUT_SHAPE[2])
d_dropout = 0.25
DCRM_OPTIMIZER = Adam(0.0001, 0.5)

GEN_OPTIMIZER = Adam(0.001, 0.5)

def load_model():
    # Checking if all the model exists
    model_names = ['DCRM', 'GEN']
    files = os.listdir(CHECKPOINT_PATH)
    for model_name in model_names:
        if model_name + ".json" not in files or \
                model_name + ".hdf5" not in files:
            print("Models not Found")
            return
    # global DCRM, GEN, COMBINED, IMAGE, GENERATED_IMAGE, CONF_GENERATED_IMAGE

    # load DCRM Model
    model_path = CHECKPOINT_PATH + "%s.json" % 'DCRM'
    weight_path = CHECKPOINT_PATH + "%s.hdf5" % 'DCRM'
    with open(model_path, 'r') as f:
        DCRM = model_from_json(f.read())
    DCRM.load_weights(weight_path)
    DCRM.compile(loss=dcrm_loss, optimizer=DCRM_OPTIMIZER)

    # load GEN Model
    model_path = CHECKPOINT_PATH + "%s.json" % 'GEN'
    weight_path = CHECKPOINT_PATH + "%s.hdf5" % 'GEN'
    with open(model_path, 'r') as f:
        GEN = model_from_json(f.read(), custom_objects={
            'InstanceNormalization': InstanceNormalization()})
    GEN.load_weights(weight_path)

    # Combined Model
    DCRM.trainable = False
    IMAGE = Input(shape=g_input_shape)
    GENERATED_IMAGE = GEN(IMAGE)
    CONF_GENERATED_IMAGE = DCRM(GENERATED_IMAGE)

    COMBINED = Model(IMAGE, [CONF_GENERATED_IMAGE, GENERATED_IMAGE])
    COMBINED.compile(loss=['mse', 'mse'], optimizer=GEN_OPTIMIZER)

    print("loaded model")
    return GEN


def get_demask_images(original_images, generated_images):
    demask_images = []
    for o_image, g_image in zip(original_images, generated_images):
        print(g_image.shape)
        width = g_image.shape[1] // 2
        x_image = g_image[:, :width]
        y_image = g_image[:, width:]
        o_image = np.concatenate((x_image,o_image, y_image), axis=1)
        demask_images.append(o_image)
    return np.asarray(demask_images)

def mask_width(img):
    image = img.copy()
    height = image.shape[0]
    width = image.shape[1]
    new_width = int(width * MASK_PERCENTAGE)
    mask = np.ones([height, new_width, 3])
    missing_x = img[:, :new_width]
    missing_y = img[:, width - new_width:]
    missing_part = np.concatenate((missing_x, missing_y), axis=1)
    image = image[:, :width - new_width]
    image = image[:, new_width:]
    return image, missing_part


def get_masked_images(images):
    mask_images = []
    missing_images = []
    for image in images:
        mask_image, missing_image = mask_width(image)
        mask_images.append(mask_image)
        missing_images.append(missing_image)
    return np.array(mask_images), np.array(missing_images)


def recursive_paint(GEN, image, factor=3):
    final_image = None
    gen_missing = None
    for i in range(factor):
        demask_image = None
        if i == 0:
            x, y = get_masked_images([image])
            gen_missing = GEN.predict(x)
            final_image = get_demask_images(x, gen_missing)[0]
        else:
            gen_missing = GEN.predict(gen_missing)
            final_image = get_demask_images([final_image], gen_missing)[0]
    return final_image


def load():
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
        gdown.download(CHECKPOINT_LINK, CHECKPOINT_PATH + 'checkpoint_24.zip')
        with zipfile.ZipFile(CHECKPOINT_PATH + 'checkpoint_24.zip', 'r') as \
                zip_ref:
            zip_ref.extractall('./')
    GEN = load_model()
    graph = tensorflow.get_default_graph()
    return GEN, graph

GEN, graph = load()

def predict(image, model):
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    GEN, graph = model
    image = image.convert('RGB')
    image = np.array(image)
    image = cv2.resize(image, (256, 256))
    cropped_image = image[:, 65:193]
    input_image = image / 127.5 - 1
    # input_image = np.expand_dims(input_image, axis=0)
    with graph.as_default():
        # predicted_image = GEN.predict(input_image)
        predicted_image = recursive_paint(GEN, input_image)
    # predicted_image = get_demask_images(input_image, predicted_image)[0]
    predicted_image = (predicted_image + 1) * 127.5
    predicted_image = predicted_image.astype(np.uint8)

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # predicted_image = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB)
    return predicted_image


INPUTS = gradio.inputs.ImageIn()
OUTPUTS = gradio.outputs.Image()
INTERFACE = gradio.Interface(fn=predict, inputs=INPUTS, 
                             outputs=OUTPUTS,
                             title='Image Outpainting', 
                             description='Restore missing parts of an image!', 
                             thumbnail='https://camo.githubusercontent.com/1374c4a783e9a1b3f31cda08e84fd1c39ebb618d/687474703a2f2f692e696d6775722e636f6d2f704455707a63592e6a7067')

INTERFACE.launch(inbrowser=True)

