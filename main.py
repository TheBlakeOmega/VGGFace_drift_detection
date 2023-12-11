from keras.utils import *
from utils import *
from model import *
import numpy as np
import random
from ConceptDriftDetectionMethod import ConceptDriftDetectionMethod
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import pickle

np.random.seed(12)
random.seed(12)
tf.random.set_seed(42)

if __name__ == '__main__':
    use_PHT = True
    detection_method_name = 'adwin'
    names = ['RobinWilliams', 'TomHulce', 'RobertoBenigni', 'TomHanks', 'LiamNeeson', 'KirstieAlley',
             'AlfreWoodard', 'StevenSeagal', 'SharonOsbourne', 'RoseanneBarr', 'RobertDavi', 'MarkHamill',
             'JohnGoodman', 'JesseVentura', 'JeffGoldblum', 'FrancesConroy', 'ColmMeaney', 'AnjelicaHuston',
             'RonHoward', 'TonyDanza', 'Unknown']
    names.sort()
    # organize_datasets(names)

    if os.path.isfile("./model.h5"):
        custom_vgg_face_model = loadModel("./model")
    else:
        train_dataset = image_dataset_from_directory("CACD2000/train", shuffle=True, batch_size=32,
                                                     image_size=(224, 224))
        custom_vgg_face_model, history = trainVGGFaceModel(train_dataset, "./model")

    stream = prepareStream()

    predictions = []
    true_labels = []
    processed_photo = []

    if not use_PHT:

        prob_model = getPredictionModel(custom_vgg_face_model)
        for person_photo_path in stream:
            photo, _ = preparePhotoTest("CACD2000/stream/" + person_photo_path)
            person_name = ''.join(person_photo_path.split("_")[1:-1])
            prediction = np.argmax(prob_model.predict(photo), axis=1)[0]
            predictions.append(names[prediction])
            true_labels.append(person_name)

    else:

        detection_method = ConceptDriftDetectionMethod(detection_method_name)
        prob_model = getPredictionModel(custom_vgg_face_model)
        for person_photo_path in stream:
            photo, original_photo = preparePhotoTest("CACD2000/stream/" + person_photo_path)
            person_name = ''.join(person_photo_path.split("_")[1:-1])
            prediction = np.argmax(prob_model.predict(photo), axis=1)[0]
            predictions.append(names[prediction])
            true_labels.append(person_name)
            processed_photo.append(photo)

            savePhoto("CACD2000/expanded_train/" + person_name + "/" + person_photo_path, original_photo)

            score = f1_score(true_labels, predictions, average='macro')
            if detection_method.test(score):
                print("Concept drift found after " + str(len(processed_photo)) + " test examples computed")
                train_dataset = image_dataset_from_directory("./CACD2000/expanded_train", shuffle=True,
                                                             batch_size=32, image_size=(224, 224))
                custom_vgg_face_model, history = updateVGGModel(train_dataset, custom_vgg_face_model)
                prob_model = getPredictionModel(custom_vgg_face_model)
                updated_scores = getFScores(prob_model, processed_photo, true_labels, names)
                detection_method.updatePHT(updated_scores)

        detection_method.closeTest()
        detection_method.makePlot()

    # STREAM ENDED
    print("Computing confusion matrix")
    test_confusion_matrix = confusion_matrix(true_labels, predictions, labels=names)
    print("Computing classification report")
    test_classification_report = classification_report(true_labels, predictions, digits=3)
    with open("results_without_detection.txt" if not use_PHT else "results_with_" + detection_method_name + ".txt", "w") as result_file:
        result_file.write("Test confusion matrix:\n")
        result_file.write(str(test_confusion_matrix) + "\n")
        result_file.write("Test classification report:\n")
        result_file.write(str(test_classification_report) + "\n")
    test_confusion_matrix_plot = ConfusionMatrixDisplay(test_confusion_matrix, display_labels=names)
    test_confusion_matrix_plot.plot()
    plt.title("Test Confusion Matrix without detection" if not use_PHT else "Test Confusion Matrix with " + detection_method_name)
    plt.savefig("test_confusion_matrix_without_detection.png" if not use_PHT else "test_confusion_matrix_with_" + detection_method_name + ".png")
    plt.close()
    with open("predictions_without_detection.pkl" if not use_PHT else "predictions_with_" + detection_method_name + ".pkl", 'wb') as file:
        pickle.dump(predictions, file)
        file.close()
    with open("results/true_labels.pkl", 'wb') as file:
        pickle.dump(true_labels, file)
        file.close()
