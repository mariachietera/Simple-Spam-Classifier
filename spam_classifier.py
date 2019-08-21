import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

DATASET_URL = "dataset.csv"  # TODO: replace with actual dataset
MX_NAME = "mx.png"

if __name__ == "__main__":
    # data loading
    print("Dataset loading..")
    data_frame = pd.read_csv(DATASET_URL)

    # data preparation
    data_frame.drop(['has_link', 'has_image'], axis=1, inplace=True)
    labels = data_frame['label'].drop_duplicates().values
    train_text, test_text, train_labels, test_labels = \
        train_test_split(data_frame["text"], data_frame["label"], test_size=.25)

    # training
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(train_text, train_labels)

    # prediction
    predicted_labels = model.predict(test_text)

    # confusion matrix
    mx = confusion_matrix(test_labels, predicted_labels, labels)
    sns.heatmap(mx.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('test label')
    plt.ylabel('predicted label')
    plt.savefig(MX_NAME)
    print("\nSee confusion matrix at {}/{}".format(os.getcwd(), MX_NAME))
    # plt.show()

    # scores
    print("\nModel evaluation scores:")
    print("Precision: {}".format(precision_score(test_labels, predicted_labels, average="weighted")))
    print("Recall: {} ".format(recall_score(test_labels, predicted_labels, average="weighted")))
    print("F-score: {}".format(f1_score(test_labels, predicted_labels, average="weighted")))
    print("Accuracy: {}\n".format(accuracy_score(test_labels, predicted_labels)))

    # live testing
    print("\nEnter text and I will label it (Q for exit):")
    new_post = input("> ")
    while new_post != "Q":
        print(model.predict([new_post])[0])
        new_post = input("> ")
