import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
import cPickle as pickle
import shap
import numpy as np
import multiprocessing
import yaml


target = "Class"
nbin = 10
dev_size = 0.5
random_seed = 1
data = pd.read_csv("creditcard.csv")
del data['Time']

# Train model
n_estimators = 10000
max_depth = 4
learning_rate = 0.1
seed = 1
tree_method = "hist"
subsample = 0.75
eval_metric = "logloss"
early_stopping = 100
ntile_calibration = 1000
ntile_exploratory = 5
test_size = 0.4
reg_alpha = 0

x = data.copy()


def bivariate():
    """
    Individual relationship between features and target
    """
    pdf = PdfPages(target + '_bivariate.pdf')
    for i in data.keys():
        if i != "Class":
            flag_frame = data[[target, i]]
            flag_frame['tile'] = pd.qcut(flag_frame[i].rank(method='first'), nbin, labels=range(1, nbin + 1))
            grouped = flag_frame.groupby('tile').mean()
            x = list(grouped[i])
            y = list(grouped[target])
            plt.figure()
            plt.plot(x, y, marker="o")
            plt.grid(True)
            plt.title(i)
            plt.ylim(0, 0.0165)
            plt.ylabel(target)
            pdf.savefig()
            plt.close()
    pdf.close()


def shapear(test):
    """
    Explain features
    """
    # Open model
    del test[target]
    with open(target + '_model.pkl', 'rb') as f:
        model = pickle.load(f)
    shap.initjs()
    shap_values = shap.TreeExplainer(model).shap_values(test)
    global_shap_vals = np.abs(shap_values).mean(0)
    global_shap_std = np.abs(shap_values).std(0)
    df = pd.DataFrame()
    df['features'] = test.columns
    df['shap'] = global_shap_vals
    df['shap_std'] = global_shap_std
    df = df.sort_values(by='shap', ascending=False)
    df.index = range(len(df))
    df.to_csv('shaps.csv')

    # Summary plot
    pdf_shap = PdfPages(target + '_shap.pdf')
    top_inds = np.argsort(-np.sum(np.abs(shap_values), 0))
    for i in top_inds:
        plt.figure()
        shap.dependence_plot(top_inds[i], shap_values,
                             test, show=False,
                             interaction_index=None,
                             alpha=0.2)
        pdf_shap.savefig()
        plt.close()
    pdf_shap.close()
    return


def train(x):
    """
    Do the training
    """
    train, test = train_test_split(x, test_size=test_size, random_state=seed)
    x_train = train[[i for i in train.keys() if i != target]]
    y_train = train[[target]]

    x_test = test[[i for i in test.keys() if i != target]]
    y_test = test[[target]]

    model = xgb.XGBClassifier(n_estimators=n_estimators,
                              max_depth=max_depth,
                              learning_rate=learning_rate,
                              seed=seed,
                              nthread=multiprocessing.cpu_count(),
                              tree_method=tree_method,
                              subsample=subsample,
                              reg_alpha=reg_alpha
                              )

    model.fit(x_train, y_train,
              eval_metric=eval_metric,
              eval_set=[(x_train, y_train), (x_test, y_test)],
              verbose=True,
              early_stopping_rounds=early_stopping)

    # Save model
    with open(target + '_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    results = model.evals_result()
    results_train = results['validation_0'][eval_metric]
    results_test = results['validation_1'][eval_metric]

    predictions = model.predict_proba(x_test)
    y_pred = [i[1] for i in predictions]

    post_model = pd.DataFrame()
    post_model['predictions'] = y_pred
    post_model[target] = list(y_test[target])
    post_model['target_tile'] = pd.qcut(post_model['predictions'].rank(method='first'),
                                        ntile_calibration, labels=range(1, ntile_calibration + 1))

    model_information(post_model, results_train, results_test)
    shapear(test)


def model_information(post_model, results_train, results_test):
    """
    Generic plots for post model analysis
    """
    scores_false = post_model['predictions'][post_model[target] == 0]
    scores_true = post_model['predictions'][post_model[target] == 1]

    pdf = PdfPages('classification_model_information.pdf')

    # Score distribution
    plt.figure()
    plt.hist(scores_true, normed=1, label="Target = 1", alpha=0.5, bins=25)
    plt.hist(scores_false, normed=1, label="Target = 0", alpha=0.5, bins=25)
    plt.legend(loc='best')
    plt.grid()
    plt.title("Score distribution test set")
    pdf.savefig()
    plt.close()

    # Plot with color
    col = list(post_model['Class'].apply(lambda x: 'r' if x == 1 else 'g'))
    plt.figure()
    plt.scatter(range(len(post_model)), list(post_model['predictions']), linewidth=0.6,
                c=col, alpha=0.5)
    plt.grid()
    plt.title("Score distribution test set")
    pdf.savefig()
    plt.close()

    # Plot learning curves
    plt.figure()
    plt.plot(results_train, label="Train")
    plt.plot(results_test, label="Test")
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel("Trees")
    plt.ylabel(eval_metric)
    plt.title("Learning curves")
    pdf.savefig()
    plt.close()
    plt.close()
    pdf.close()

    grouped = post_model.groupby('target_tile').mean()
    suma = post_model.groupby('target_tile').sum()[target]
    grouped['total'] = suma
    grouped['total_bucket'] = list(post_model.groupby('target_tile').count()[target])
    grouped.to_csv('calibration.csv', index=False)


def predict(extractor):
#    """
#    Get a dictionary like:
#     extractor = {
#         'V1': 4,
#         'V2': 1,
#         'V3': 1,
#         'V4': 1,
#         'V5': 2,
#         'V6': 1,
#         'V7': 1,
#         'V8': 1,
#         'V9': 1,
#         'V10': 1,
#         'V11': 1,
#         'V12': 1,
#         'V13': 1,
#         'V14': 1,
#         'V15': 1,
#         'V16': 1,
#         'V17': 1,
#         'V18': 1,
#         'V19': 1,
#         'V20': 1,
#         'V21': 1,
#         'V22': 1,
#         'V23': 1,
#         'V24': 1,
#         'V25': 1,
#         'V26': 1,
#         'V27': 1,
#         'V28': 1,
#         'Amount': 2
#     }
#    Return probability
#    """
# Load cross validation parameters
    with open("columns.config",
              "r") as f:
        columns = yaml.load(f)
    df = pd.DataFrame(eval(extractor), index=[0])[columns['cols']]
    with open(target + '_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model.predict_proba(df)[0][1]



if __name__ == "__main__":
    bivariate()
    train(data)

