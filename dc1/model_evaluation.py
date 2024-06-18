import torch
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from collections import Counter
from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset
from pathlib import Path

from dc1.improved_net_256 import ImprovedNet_256
from dc1.improved_net_128 import ImprovedNet_128
from dc1.improved_net_8_to_64 import ImprovedNet_8_64
from dc1.improved_net_16_to_64 import ImprovedNet_16_64
from dc1.net import Net

class_mapping = {
    0: 'Atelectasis',
    1: 'Effusion',
    2: 'Infiltration',
    3: 'No finding',
    4: 'Nodule',
    5: 'Pneumothorax'
}

class_to_weight = {
    'Effusion': 5,
    'Nodule': 4,
    'Atelectasis': 3,
    'Pneumothorax': 2,
    'Infiltration': 1,
    'No finding': 0,
}


def evaluate_model(path, trained_model, batch_size):
    test_dataset = ImageDataset(Path("data/X_test.npy"), Path("data/Y_test.npy"))

    # Load the model
    model = trained_model  # met a model with 6 possible classes
    model.eval()  # Set the model to evaluation mode

    # Load the model weights
    if torch.cuda.is_available():
        state_dict = torch.load(path)
    elif torch.backends.mps.is_available():  # PyTorch supports Apple Silicon GPU's from version 1.12
        state_dict = torch.load(path, map_location=torch.device('mps'))
    else:
        state_dict = torch.load(path, map_location=torch.device('cpu'))

    model.load_state_dict(state_dict)

    predictions = []
    probability_list = []
    actuals = []

    test_sampler = BatchSampler(batch_size=batch_size, dataset=test_dataset, balanced=True)

    # Iterate over the test set
    for picture_batch, target_batch in test_sampler:
        # Preprocess the input
        with torch.no_grad():
            # Make a prediction
            y_pred = model(picture_batch).squeeze()
            probabilities = torch.softmax(y_pred, dim=1)
            predicted_category = torch.argmax(probabilities, dim=1)

        # Store the predicted and actual category
        probabilities = probabilities.tolist()
        probability_list.extend(probabilities)
        predicted_category = predicted_category.tolist()
        predictions.extend(predicted_category)
        target_batch = target_batch.tolist()
        actuals.extend(target_batch)

    # Create a DataFrame with the results
    resulting_df = pd.DataFrame({
        "Predicted Category": predictions,
        "Actual Category": actuals
    })

    correct_df = resulting_df[resulting_df['Predicted Category'] == resulting_df['Actual Category']]

    probability_df = pd.DataFrame({
        'Actuals': actuals,
        'Probabilities': probability_list
    })

    len_correct = len(correct_df)
    total_len = len(resulting_df)

    accuracy = len_correct / total_len
    print(len_correct, total_len)
    print(f'accuracy: {accuracy}')

    resulting_df['Actual Category'] = resulting_df['Actual Category'].map(class_mapping)
    resulting_df['Predicted Category'] = resulting_df['Predicted Category'].map(class_mapping)
    return resulting_df, probability_df


def map_categories(results):
    results['Actual Category'] = results['Actual Category'].map(class_to_weight)
    results['Predicted Category'] = results['Predicted Category'].map(class_to_weight)
    return results


def plot_confusion_matrix(df_values, model_path):
    class_names = list(class_to_weight.keys())
    cm = confusion_matrix(df_values['Actual Category'], df_values['Predicted Category'], labels=class_names)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(model_path)
    plt.xlabel('Predicted Categories')
    plt.ylabel('Actual Categories')
    plt.show()


def plot_tp_confusion_matrix(df_values, model_path):
    df_to_augmet = df_values.copy()
    df_to_augmet = map_categories(df_to_augmet)

    result = []
    for p, a in zip(df_to_augmet['Predicted Category'], df_to_augmet['Actual Category']):
        if p == a and p == 0:
            result.append('True Negative')
        elif p == a and p != 0:
            result.append('True Positive')
        elif p > a:
            result.append('False Positive')
        elif p < a and p == 0:
            result.append('False Negative')
        elif p < a and p != 0:
            result.append('False Negative')

    count_dict = Counter(result)
    TP = count_dict['True Positive']
    FP = count_dict['False Positive']
    FN = count_dict['False Negative']
    TN = count_dict['True Negative']

    data = [[TN, FP], [FN, TP]]
    plt.figure(figsize=(10, 7))
    # change the cmap if you would like the heatmap to have a different color
    sns.heatmap(data, annot=True, fmt='d', cmap='Blues')

    # Set custom labels for the x-axis and y-axis
    plt.xticks(ticks=[0.5, 1.5], labels=['Predicted Negative', 'Predicted Positive'])
    plt.yticks(ticks=[0.5, 1.5], labels=['Actual Negative', 'Actual Positive'])

    plt.title(model_path)
    plt.xlabel('Predicted Categories')
    plt.ylabel('Actual Categories')
    plt.show()

    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    F1_score = (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1_score:", F1_score)


def plot_one_model_roc_curves(df_probabilities):
    plt.figure(figsize=(20, 14))

    actual_labels = df_probabilities['Actuals']
    prediction_chances = df_probabilities.drop(columns=['Actuals']).values

    for i, illness in class_mapping.items():
        # Binary classification for the current illness vs. all others
        actual_binary = (actual_labels == i).astype(int)
        actual_binary = actual_binary.values

        # print(prediction_chances)
        prediction_scores = [chance[0] for chance in prediction_chances]  # Convert to numpy array
        prediction_list_i = []
        for prediction_list in prediction_scores:
            prediction_list_i.extend([prediction_list[i]])
        print(prediction_list_i)
        print(actual_binary)

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(actual_binary, prediction_list_i)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.plot(fpr, tpr, label='%s (AUC = %0.2f)' % (illness, roc_auc))

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


def model_evaluation(model_path, model_train, batchsize):
    results_df, probability_df = evaluate_model(path=model_path, trained_model=model_train, batch_size=batchsize)

    plot_confusion_matrix(results_df, model_path)

    plot_one_model_roc_curves(probability_df)

    plot_tp_confusion_matrix(results_df, model_path)


if __name__ == "__main__":
    model_path = 'model_weights/model_improved_net_128_final.txt'
    model_train = ImprovedNet_128(n_classes=6)
    batch_size = 100

    model_evaluation(model_path, model_train, batch_size)
