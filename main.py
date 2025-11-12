import pandas as pd
#from src.classifyre import classify_with_classyfire
#from src.big_file import big_file_classifier
from src.evaluation import save_results_histogram
from src.histogram_and_classes import get_class_dict_and_histogram
from src.training_functions import training
from src_elena.sequentialTraining_module import *

if __name__ == "__main__":
    fp = os.path.join("resources", "smrt_fingerprints.csv")
    model = os.path.join("results_elena", "model.h5")
    scaler = os.path.join("results_elena", "scaler.pkl")
    train_and_test(fp, model, scaler)

"""
if __name__ == "__main__":

    # Parameters:
    train_using_classes = True

    # classify_with_classyfire()
    # big_file_classifier()

    if train_using_classes:
        class_dict = get_class_dict_and_histogram(300)
    else:
        # Train with all smrt:
        df = pd.read_csv('tests_for_smrt/resources/smrt_fingerprints.csv')
        class_dict = {'all': (len(df['inchi']), df['inchi'].tolist())}

    results_df = training(class_dict)
    print(results_df)
    save_results_histogram(results_df)

    #classification_elena()
    #training_elena()
"""
