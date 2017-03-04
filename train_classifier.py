import glob
import time
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
from sklearn.cross_validation import train_test_split
from library import extract_features


def train_classifier(color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat):

    # Reads in cars and notcars
    cars = glob.glob('training_images/vehicles/*.png')
    notcars = glob.glob('training_images/non_vehicles/*.png')
    supplements = glob.glob('training_images/supplements/*.png')


    car_features = extract_features(cars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
    supplement_features = extract_features(supplements, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)

    # Shuffles features
    car_features = np.array(car_features)
    notcar_features = np.array(notcar_features)
    supplement_features = np.array(supplement_features)
    np.random.shuffle(notcar_features)
    np.random.shuffle(car_features)


    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Fits a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Applies the scaler to X
    scaled_X = X_scaler.transform(X)

    # Defines the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Splits up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    X_train = np.vstack([X_train, supplement_features])
    y_train = np.append(y_train, np.ones(len(supplement_features)))

    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))

    svc = SVC(probability=True)

    # Checks the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Checks the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Checks the prediction time for a single sample
    t = time.time()

    # Writes to disk
    write_obj = {"X_scaler": X_scaler, "svc": svc}
    with open('model.p', 'wb') as file:
        pickle.dump(write_obj, file)