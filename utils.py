import os
from operator import itemgetter
from matplotlib.image import imread
import numpy as np
import mtcnn
import cv2
import random
from sklearn.metrics import f1_score

face_detector = mtcnn.MTCNN()


def organize_datasets(selected_names):
    imgList = [x for x in os.listdir("CACD2000/CACD2000/") if ".jpg" in x]
    dict_of_lists = {}
    for file_path in imgList:
        label = ''.join(file_path.split("_")[1:-1])
        if label not in dict_of_lists:
            dict_of_lists[label] = []
        dict_of_lists[label].append(file_path)

    max_people = 50
    processed = 0
    unknown_people_processed = 0
    os.mkdir("./CACD2000/train/unknown")
    for name in dict_of_lists:
        print("Organizing " + name)
        if name in selected_names:
            os.mkdir("./CACD2000/train/" + str(name))
            tran_img_number = int(len(dict_of_lists[name]) / 3)
            ordered_list = sorted(dict_of_lists[name], key=lambda x: itemgetter(0)(x.split("_")))
            for i in range(tran_img_number):
                photo = preparePhotoTrain("CACD2000/CACD2000/" + ordered_list[i])
                if photo is not None:
                    savePhoto("CACD2000/train/" + name + "/" + ordered_list[i], photo)
            for i in range(tran_img_number, len(dict_of_lists[name])):
                photo = preparePhotoTrain("CACD2000/CACD2000/" + ordered_list[i])
                if photo is not None:
                    savePhoto("CACD2000/stream/" + ordered_list[i], photo)
            processed += 1
            if processed == max_people:
                break
        else:
            unknown_people_processed += 1
            if unknown_people_processed % 19 == 0:
                random_image = random.choice(dict_of_lists[name])
                photo = preparePhotoTrain("CACD2000/CACD2000/" + random_image)
                if photo is not None:
                    savePhoto("CACD2000/train/unknown/" + random_image, photo)


def preparePhotoTrain(path):
    global face_detector
    photo = imread(path)
    face_detected = face_detector.detect_faces(photo)
    if len(face_detected) != 0:
        x1, y1, width, length = face_detected[0]['box']
        x2, y2 = x1 + width, y1 + length
        cropped_photo = photo[y1:y2, x1:x2]
        cropped_photo = cv2.resize(cropped_photo, (224, 224), interpolation=cv2.INTER_AREA)
        return cropped_photo
    else:
        return None


def preparePhotoTest(path):
    test_image = imread(path)
    image_array = np.array(test_image)
    return np.expand_dims(image_array, axis=0), test_image


def prepareStream():
    imgList = [x for x in os.listdir("CACD2000/stream/") if ".jpg" in x]
    dict_of_lists = {}
    stream = []
    for file_path in imgList:
        age = file_path.split("_")[0]
        if age not in dict_of_lists:
            dict_of_lists[age] = []
        dict_of_lists[age].append(file_path)
    for age in dict_of_lists:
        random.shuffle(dict_of_lists[age])
        stream.extend(dict_of_lists[age])
    return stream


def savePhoto(save_path, photo):
    cv2.imwrite(save_path, cv2.cvtColor(photo, cv2.COLOR_RGB2BGR))


def getFScores(model, photos, true_labels, set_labels):
    predictions = []
    scores = []
    for photo in photos:
        prediction = np.argmax(model.predict(photo), axis=1)[0]
        predictions.append(set_labels[prediction])
    for n in range(len(photos)):
        scores.append(f1_score(true_labels[:(n + 1)], predictions[:(n + 1)], average='macro'))
    return scores
