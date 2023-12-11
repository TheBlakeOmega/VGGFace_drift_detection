from keras import Sequential, Model
from keras.models import load_model
from keras.losses import SparseCategoricalCrossentropy
from keras.layers import RandomFlip, RandomRotation, Flatten, Dense, Softmax
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from keras_vggface import VGGFace


def trainVGGFaceModel(train_dataset, save_file_name=None):
    data_augmentation = Sequential([
        RandomFlip('horizontal'),
        RandomRotation(0.2)
    ])

    model = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3))
    model.trainable = False
    # last_layer = model.get_layer('avg_pool').output
    num_people = 21
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = model(x)
    x = Flatten(name='flatten')(x)
    out = Dense(num_people, name='classifier')(x)
    custom_vgg_face_model = Model(inputs, out)

    custom_vgg_face_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                                  loss=SparseCategoricalCrossentropy(from_logits=True),
                                  metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='loss', patience=10)
    model_checkpoint = ModelCheckpoint(filepath=save_file_name + '.h5', monitor='loss', save_best_only=True)
    history = custom_vgg_face_model.fit(train_dataset, epochs=150, callbacks=[early_stop, model_checkpoint])

    return custom_vgg_face_model, history


def updateVGGModel(updated_train_dataset, model):
    early_stop = EarlyStopping(monitor='loss', patience=10)
    history = model.fit(updated_train_dataset, epochs=150, callbacks=[early_stop])
    return model, history


def getPredictionModel(custom_vgg_face_model):
    prob_model = Sequential([
        custom_vgg_face_model,
        tf.keras.layers.Softmax()
    ])
    return prob_model


def loadModel(save_file_name):
    """json_file = open(save_file_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    custom_vgg_face_model = model_from_json(loaded_model_json)
    custom_vgg_face_model.load_weights(save_file_name + ".h5")
    custom_vgg_face_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                                  loss=SparseCategoricalCrossentropy(from_logits=True),
                                  metrics=['accuracy'])"""
    custom_vgg_face_model = load_model(save_file_name + ".h5")
    return custom_vgg_face_model
