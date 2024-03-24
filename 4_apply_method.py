import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, Input
from keras.optimizers import Adam, RMSprop, Nadam, SGD, Adagrad, Adadelta, Adamax
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from keras.applications.vgg19 import VGG19, preprocess_input
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import InceptionV3

from utils.constants import championsList
from utils.constants import champion_id_map
from config import pro, ranked
from utils.tools import showCurrentTime

batch_size = 64
nbr_epochs = 500
learning_rate = 0.001
random_state = 42

# for knn
n_neighbors = 5

# for random_forest
n_estimators = 100

# for mlp and sgd
max_iter = nbr_epochs

def knn(X_train, y_train, X_test, y_test, n_neighbors):

    # Create and train the KNN model
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate the model's accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Accuracy: {accuracy}')

    return model, accuracy, y_pred, y_test

def random_forest(X_train, y_train, X_test, y_test, n_estimators, random_state):

    # Create and train the Random Forest model
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate the model's accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Accuracy: {accuracy}')

    return model, accuracy, y_pred, y_test

def mlp(X_train, y_train, X_test, y_test, max_iter, random_state):

    # Create and train the MLP model
    model = MLPClassifier(hidden_layer_sizes=(X_train.shape[1],), max_iter=max_iter, random_state=random_state)

    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate the model's accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Accuracy: {accuracy}')

    return model, accuracy, y_pred, y_test

import math
def step_decay(epoch):
    initial_lrate = learning_rate
    drop = 0.8
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def neural_network(X_train, y_train, X_test, y_test, l2_regularization=0.01):

    # Build the neural network model
    model = Sequential([
        Dense(256, activation='relu',  input_shape=(X_train.shape[1],), kernel_regularizer=l2(l2_regularization)),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    # SGD, Adagrad, Adadelta have bad scores
    # Adam, RMSprop, Nadam, , Adamax
    # Adamax less good than Adam
    # RMS prop almost same as Adam
    # Nadam a little bit less good than Adam and RMSProp
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    # to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

    # Define the learning rate scheduler
    lrate = LearningRateScheduler(step_decay)
    callbacks = [early_stopping]#, lrate]

    # Train the model
    model.fit(X_train, y_train, epochs=nbr_epochs, batch_size=batch_size, validation_split=0.2, callbacks=callbacks, verbose=0)

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy}')

    y_pred = model.predict(X_test)

    return model, accuracy, y_pred, y_test

# https://keras.io/examples/time_series/time_series_classification_from_scratch/
from keras.layers import Conv1D, BatchNormalization, ReLU, GlobalAveragePooling1D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Bidirectional, LSTM, GlobalMaxPooling1D
def time_series_classification_from_scratch_model(X_train, y_train, X_test, y_test):

    X_train = np.array(X_train).reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))

    input_layer = Input(shape=(X_train.shape[1], 1))

    conv1 = Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)

    conv2 = Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)

    conv3 = Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = ReLU()(conv3)

    lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(conv3)

    gap = GlobalAveragePooling1D()(lstm_layer)

    output_layer = Dense(1, activation="sigmoid")(gap)

    model = Model(inputs=input_layer, outputs=output_layer)

    callbacks = [
        # ModelCheckpoint(
        #     "best_model.keras", save_best_only=True, monitor="val_loss"
        # ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=0.0001
        ),
        EarlyStopping(monitor="val_loss", patience=10, verbose=1),
    ]
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=nbr_epochs,
        callbacks=callbacks,
        validation_split=0.2,
        verbose=0,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test)

    y_pred = model.predict(X_test)

    print("Test accuracy", test_acc)
    print("Test loss", test_loss)

    return model, test_acc, y_pred, y_test


def svm(X_train, y_train, X_test, y_test, random_state):

    # Create and train the SVM model
    model = SVC(kernel='rbf', C=1.0, random_state=random_state)

    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate the model's accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Accuracy: {accuracy}')

    return model, accuracy, y_pred, y_test

def sgd(X_train, y_train, X_test, y_test, max_iter, random_state):

    # Create and train the SGD model
    model = SGDClassifier(loss='log_loss', max_iter=max_iter, random_state=random_state)

    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate the model's accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Accuracy: {accuracy}')

    return model, accuracy, y_pred, y_test

def gbt(X_train, y_train, X_test, y_test, n_estimators=100, learning_rate=0.1, random_state=42):

    # Create and train the GBT model
    model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)

    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate the model's accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Accuracy: {accuracy}')

    return model, accuracy, y_pred, y_test

def apply_method(method_name, game_type, fold_used):
    """
    method to apply a specific model on a specific fold
    """

    print("Method",method_name)

    input_path = "data/"+game_type+"/fold"+str(fold_used)

    # import features vector for the model
    test_data = pd.read_csv(input_path+"/test/feature_vectors.csv")
    train_data = pd.read_csv(input_path+"/train/feature_vectors.csv")

    X_test = test_data.drop(columns=['winner'], errors='ignore')
    X_train = train_data.drop(columns=['winner'], errors='ignore')

    # blue is 1, else (red) it's 0
    y_test = test_data['winner']
    y_train = train_data['winner']

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    match method_name:

        case "knn":
            return knn(X_train, y_train, X_test, y_test, n_neighbors)
        
        case "random_forest":
            return random_forest(X_train, y_train, X_test, y_test, n_estimators, random_state)
        
        case "mlp":
            return mlp(X_train, y_train, X_test, y_test, max_iter, random_state)
        
        case "neural_network":
            return neural_network(X_train, y_train, X_test, y_test)
        
        case "svm":
            return svm(X_train, y_train, X_test, y_test, random_state)
        
        case "sgd":
            return sgd(X_train, y_train, X_test, y_test, max_iter, random_state)
        
        case "gbt":
            return gbt(X_train, y_train, X_test, y_test, n_estimators, learning_rate, random_state)
        
        case "time_series_classification":
            return time_series_classification_from_scratch_model(X_train, y_train, X_test, y_test)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
def generate_confusion_matrix(y_true, y_pred, model_name):

    # Convert probabilities to binary predictions using a threshold (e.g., 0.5)
    threshold = 0.5
    y_pred = (y_pred > threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)

    # Create "confusion_matrix" folder if it does not exist
    save_folder = "confusion_matrices"
    os.makedirs(save_folder, exist_ok=True)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    if game_type == "":
        title = f'Confusion Matrix\n{model_name}'
    else:
        title = f'{game_type.title()} Confusion Matrix\n{model_name}'
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Save the image in "confusion_matrix" folder
    save_path = os.path.join(save_folder, f'{model_name}.pdf')
    plt.savefig(save_path)

    # Close the figure to prevent memory issues
    plt.close()

    return cm

def generate_all_folds_confusion_matrix(all_folds_cm, method_name):

    all_folds_cm = all_folds_cm.astype(int)

    # Plot the all folds confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(all_folds_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.title(method_name.title()+'\nConfusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Save the all folds confusion matrix image in "confusion_matrices" folder
    save_folder = "confusion_matrices"
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f'{method_name}.pdf')
    plt.savefig(save_path)

    # Close the figure to prevent memory issues
    plt.close()


game_types = []
if pro:
    game_types.append("pro")
if ranked:
    game_types.append("ranked")

nbr_of_folds = 5

knn_models = []
random_forest_models = []
mlp_models = []
neural_network_models = []
svm_models = []
sgd_models = []
time_series_models = []

for game_type in game_types:

    knn_accuracies = []
    random_forest_accuracies = []
    mlp_accuracies = []
    neural_network_accuracies = []
    svm_accuracies = []
    sgd_accuracies = []
    time_series_accuracies = []

    knn_all_folds_cm = np.zeros((2, 2))
    random_forest_all_folds_cm = np.zeros((2, 2))
    mlp_all_folds_cm = np.zeros((2, 2))
    neural_network_all_folds_cm = np.zeros((2, 2))
    svm_all_folds_cm = np.zeros((2, 2))
    sgd_all_folds_cm = np.zeros((2, 2))
    time_series_all_folds_cm = np.zeros((2, 2))

    for i in range(nbr_of_folds):

        print(f"{showCurrentTime()}{game_type} FOLD n°{i+1}:")

        # Apply knn model
        knn_model, knn_accuracy, knn_pred, y_test_knn = apply_method("knn", game_type, i + 1)
        knn_accuracies.append(knn_accuracy)
        knn_all_folds_cm += generate_confusion_matrix(y_test_knn, knn_pred, f'{game_type} k-NN fold {i+1}')
        knn_models.append(knn_model)

        # Apply random forest model
        random_forest_model , random_forest_accuracy, random_forest_pred, y_test_rf = apply_method("random_forest", game_type, i + 1)
        random_forest_accuracies.append(random_forest_accuracy)
        random_forest_all_folds_cm += generate_confusion_matrix(y_test_rf, random_forest_pred, f'{game_type} Random Forest fold {i+1}')
        random_forest_models.append(random_forest_model)

        # Apply mlp model
        mlp_model, mlp_accuracy, mlp_pred, y_test_mlp = apply_method("mlp", game_type, i + 1)
        mlp_accuracies.append(mlp_accuracy)
        mlp_all_folds_cm += generate_confusion_matrix(y_test_mlp, mlp_pred, f'{game_type} MLP fold{i+1}')
        mlp_models.append(mlp_model)

        # Apply neural network model
        neural_network_model, neural_network_accuracy, neural_network_pred, y_test_nn = apply_method("neural_network", game_type, i + 1)
        neural_network_accuracies.append(neural_network_accuracy)
        neural_network_all_folds_cm += generate_confusion_matrix(y_test_nn, neural_network_pred, f'{game_type} Neural Network fold {i+1}')
        neural_network_models.append(neural_network_model)

        # Apply svm model
        svm_model, svm_accuracy, svm_pred, y_test_svm = apply_method("svm", game_type, i + 1)
        svm_accuracies.append(svm_accuracy)
        svm_all_folds_cm += generate_confusion_matrix(y_test_svm, svm_pred, f'{game_type} SVM fold {i+1}')
        svm_models.append(svm_model)

        # Apply sgd model
        sgd_model, sgd_accuracy, sgd_pred, y_test_sgd = apply_method("sgd", game_type, i + 1)
        sgd_accuracies.append(sgd_accuracy)
        sgd_all_folds_cm += generate_confusion_matrix(y_test_sgd, sgd_pred, f'{game_type} SGD fold {i+1}')
        sgd_models.append(sgd_model)

        # Apply time series classification model
        time_series_model, time_series_accuracy, time_series_pred, y_test_ts = apply_method("time_series_classification", game_type, i + 1)
        time_series_accuracies.append(time_series_accuracy)
        time_series_all_folds_cm += generate_confusion_matrix(y_test_ts, time_series_pred, f'{game_type} Time Series fold {i+1}')
        time_series_models.append(time_series_model)

        print(f"-----\n{showCurrentTime()}FINISHED {game_type} FOLD n°{i+1}\n-----")

    # Merge the results
    average_knn_accuracy = np.mean(knn_accuracies)
    average_random_forest_accuracy = np.mean(random_forest_accuracies)
    average_mlp_accuracy = np.mean(mlp_accuracies)
    average_neural_network_accuracy = np.mean(neural_network_accuracies)
    average_svm_accuracy = np.mean(svm_accuracies)
    average_sgd_accuracy = np.mean(sgd_accuracies)
    average_time_series_classification = np.mean(time_series_accuracies)

    # Calculate standard deviation for each model
    knn_std = np.std(knn_accuracies)
    random_forest_std = np.std(random_forest_accuracies)
    mlp_std = np.std(mlp_accuracies)
    neural_network_std = np.std(neural_network_accuracies)
    svm_std = np.std(svm_accuracies)
    sgd_std = np.std(sgd_accuracies)
    time_series_std = np.std(time_series_accuracies)

    print(f'Average Test Accuracy for k-NN across all folds: {average_knn_accuracy}')
    print(f'Average Test Accuracy for Random Forest across all folds: {average_random_forest_accuracy}')
    print(f'Average Test Accuracy for MLP across all folds: {average_mlp_accuracy}')
    print(f'Average Test Accuracy for Neural Network across all folds: {average_neural_network_accuracy}')
    print(f'Average Test Accuracy for SVM across all folds: {average_svm_accuracy}')
    print(f'Average Test Accuracy for SGD across all folds: {average_sgd_accuracy}')
    print(f'Average Test Accuracy for time series classification from scratch across all folds: {average_time_series_classification}')

    # Print standard deviations
    print(f'Standard Deviation for k-NN: {knn_std}')
    print(f'Standard Deviation for Random Forest: {random_forest_std}')
    print(f'Standard Deviation for MLP: {mlp_std}')
    print(f'Standard Deviation for Neural Network: {neural_network_std}')
    print(f'Standard Deviation for SVM: {svm_std}')
    print(f'Standard Deviation for SGD: {sgd_std}')
    print(f'Standard Deviation for Time Series: {time_series_std}')

    # Plot the All folds confusion matrix for each model
    generate_all_folds_confusion_matrix(knn_all_folds_cm, f'{game_type} All folds k-NN')
    generate_all_folds_confusion_matrix(random_forest_all_folds_cm, f'{game_type} All folds Random Forest')
    generate_all_folds_confusion_matrix(mlp_all_folds_cm, f'{game_type} All folds MLP')
    generate_all_folds_confusion_matrix(neural_network_all_folds_cm, f'{game_type} All folds Neural Network')
    generate_all_folds_confusion_matrix(svm_all_folds_cm, f'{game_type} All folds SVM')
    generate_all_folds_confusion_matrix(sgd_all_folds_cm, f'{game_type} All folds SGD')
    generate_all_folds_confusion_matrix(time_series_all_folds_cm, f'{game_type} All folds Time Series')

    print(showCurrentTime()+f"{game_type} DONE !")

print(showCurrentTime()+"let's train model with ranked games and use pro games as test data")
game_type = ""
knn_accuracies = []
random_forest_accuracies = []
mlp_accuracies = []
neural_network_accuracies = []
svm_accuracies = []
sgd_accuracies = []
time_series_accuracies = []

knn_all_folds_cm = np.zeros((2, 2))
random_forest_all_folds_cm = np.zeros((2, 2))
mlp_all_folds_cm = np.zeros((2, 2))
neural_network_all_folds_cm = np.zeros((2, 2))
svm_all_folds_cm = np.zeros((2, 2))
sgd_all_folds_cm = np.zeros((2, 2))
time_series_all_folds_cm = np.zeros((2, 2))

# first 5 are pro, then next 5 models are ranked one
for i in range(nbr_of_folds):

    print(f"{showCurrentTime()} FOLD n°{i+1}:")

    # import test data from pro at current split
    input_path = "data/pro/fold"+str(i+1)
    test_data = pd.read_csv(input_path+"/test/feature_vectors.csv")
    X_test = test_data.drop(columns=['winner'], errors='ignore')
    y_test = test_data['winner']

    # Apply knn model
    knn_pred = knn_models[i+5].predict(X_test)
    knn_accuracy = accuracy_score(y_test, knn_pred)
    knn_accuracies.append(knn_accuracy)
    knn_all_folds_cm += generate_confusion_matrix(y_test, knn_pred, f'k-NN fold {i+1}\nranked as train - pro as test')

    # Apply random forest model
    random_forest_pred = random_forest_models[i+5].predict(X_test)
    random_forest_accuracy = accuracy_score(y_test, random_forest_pred)
    random_forest_accuracies.append(random_forest_accuracy)
    random_forest_all_folds_cm += generate_confusion_matrix(y_test, random_forest_pred, f'Random Forest fold {i+1}\nranked as train - pro as test')

    # Apply mlp model
    mlp_pred = mlp_models[i+5].predict(X_test)
    mlp_accuracy = accuracy_score(y_test, mlp_pred)
    mlp_accuracies.append(mlp_accuracy)
    mlp_all_folds_cm += generate_confusion_matrix(y_test, mlp_pred, f'MLP fold {i+1}\nranked as train - pro as test')

    # Apply neural network model
    neural_network_pred = neural_network_models[i+5].predict(X_test)
    threshold = 0.5
    neural_network_pred = (neural_network_pred > threshold).astype(int)
    neural_network_accuracy = accuracy_score(y_test, neural_network_pred)
    neural_network_accuracies.append(neural_network_accuracy)
    neural_network_all_folds_cm += generate_confusion_matrix(y_test, neural_network_pred, f'Neural Network fold {i+1}\nranked as train - pro as test')

    # Apply svm model
    svm_pred = svm_models[i+5].predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    svm_accuracies.append(svm_accuracy)
    svm_all_folds_cm += generate_confusion_matrix(y_test, svm_pred, f'SVM fold {i+1}\nranked as train - pro as test')

    # Apply sgd model
    sgd_pred = sgd_models[i+5].predict(X_test)
    sgd_accuracy = accuracy_score(y_test, sgd_pred)
    sgd_accuracies.append(sgd_accuracy)
    sgd_all_folds_cm += generate_confusion_matrix(y_test, sgd_pred, f'SGD fold {i+1}\nranked as train - pro as test')

    # Apply time series classification model
    time_series_pred = time_series_models[i+5].predict(X_test)
    threshold = 0.5
    time_series_pred = (time_series_pred > threshold).astype(int)
    time_series_accuracy = accuracy_score(y_test, time_series_pred)
    time_series_accuracies.append(time_series_accuracy)
    time_series_all_folds_cm += generate_confusion_matrix(y_test, time_series_pred, f'Time Series fold {i+1}\nranked as train - pro as test')

    print(f"-----\n{showCurrentTime()}FINISHED {game_type} FOLD n°{i+1}\n-----")

# Merge the results
average_knn_accuracy = np.mean(knn_accuracies)
average_random_forest_accuracy = np.mean(random_forest_accuracies)
average_mlp_accuracy = np.mean(mlp_accuracies)
average_neural_network_accuracy = np.mean(neural_network_accuracies)
average_svm_accuracy = np.mean(svm_accuracies)
average_sgd_accuracy = np.mean(sgd_accuracies)
average_time_series_classification = np.mean(time_series_accuracies)

# Calculate standard deviation for each model
knn_std = np.std(knn_accuracies)
random_forest_std = np.std(random_forest_accuracies)
mlp_std = np.std(mlp_accuracies)
neural_network_std = np.std(neural_network_accuracies)
svm_std = np.std(svm_accuracies)
sgd_std = np.std(sgd_accuracies)
time_series_std = np.std(time_series_accuracies)

print(f'Average Test Accuracy for k-NN across all folds (ranked as train / pro as test): {average_knn_accuracy}')
print(f'Average Test Accuracy for Random Forest across all folds (ranked as train / pro as test): {average_random_forest_accuracy}')
print(f'Average Test Accuracy for MLP across all folds (ranked as train / pro as test): {average_mlp_accuracy}')
print(f'Average Test Accuracy for Neural Network across all folds (ranked as train / pro as test): {average_neural_network_accuracy}')
print(f'Average Test Accuracy for SVM across all folds (ranked as train / pro as test): {average_svm_accuracy}')
print(f'Average Test Accuracy for SGD across all folds (ranked as train / pro as test): {average_sgd_accuracy}')
print(f'Average Test Accuracy for Time Series classification from scratch across all folds (ranked as train / pro as test): {average_time_series_classification}')

# Print standard deviations
print(f'Standard Deviation for k-NN (ranked as train / pro as test): {knn_std}')
print(f'Standard Deviation for Random Forest (ranked as train / pro as test): {random_forest_std}')
print(f'Standard Deviation for MLP (ranked as train / pro as test): {mlp_std}')
print(f'Standard Deviation for Neural Network (ranked as train / pro as test): {neural_network_std}')
print(f'Standard Deviation for SVM (ranked as train / pro as test): {svm_std}')
print(f'Standard Deviation for SGD (ranked as train / pro as test): {sgd_std}')
print(f'Standard Deviation for Time Series: {time_series_std}')

# Plot the All folds confusion matrix for each model
generate_all_folds_confusion_matrix(knn_all_folds_cm, f'All folds k-NN\nranked as train - pro as test')
generate_all_folds_confusion_matrix(random_forest_all_folds_cm, f'All folds Random Forest\nranked as train - pro as test')
generate_all_folds_confusion_matrix(mlp_all_folds_cm, f'All folds MLP\nranked as train - pro as test')
generate_all_folds_confusion_matrix(neural_network_all_folds_cm, f'All folds Neural Network\nranked as train - pro as test')
generate_all_folds_confusion_matrix(svm_all_folds_cm, f'All folds SVM\nranked as train - pro as test')
generate_all_folds_confusion_matrix(sgd_all_folds_cm, f'All folds SGD\nranked as train - pro as test')
generate_all_folds_confusion_matrix(time_series_all_folds_cm, f'All folds Time Series\nranked as train - pro as test')

print(showCurrentTime()+"DONE !")