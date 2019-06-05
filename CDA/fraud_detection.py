import numpy as np
import pandas
import matplotlib
import matplotlib.pyplot as plt

# read data

src = "./data_for_student_case.csv"
df = pandas.read_csv(src)

# Refused label exist in most credit card with fraud history
# pandas.set_option('display.max_columns', None)
# frauded_users=df[df["simple_journal"]=="Chargeback"]["card_id"].unique()
# print(df[df["card_id"].isin(frauded_users)])
# print(df["currencycode"].unique())
print("data size before process " + str(len(df)))
# print(df.head()["amount"])
# data processing
df = df[df["simple_journal"] != "Refused"]  # remove data with refused label
print("data size after process " + str(len(df)))


# convert currency
def convertCurrency(x):
    currency = {'MXN': 0.0469002, 'AUD': 0.626700, 'NZD': 0.588162, 'GBP': 1.16682, 'SEK': 0.0931781}
    amount = x["amount"]
    currencycode = x["currencycode"]
    return round(amount * currency[currencycode])


df.loc[:, "converted_amount"] = df.apply(lambda x: convertCurrency(x), axis=1)
# print(df["converted_amount"].head())

# barchart
import seaborn as sns

sns.set(style="whitegrid")
tips = sns.barplot(x="currencycode", y="converted_amount", hue="simple_journal", data=df)
tips.set(xlabel='currencycode', ylabel='Money')
plt.show()
tips.get_figure().savefig("barchart")
# boxplot
tips = sns.boxplot(x="currencycode", y="converted_amount", hue="simple_journal", data=df)
tips.set(xlabel='Labels', ylabel='Money')
plt.show()
tips.get_figure().savefig("box")
# kernel density estimation
tips = sns.distplot(df[df["simple_journal"] == "Chargeback"]['amount'], hist=False)
tips = sns.distplot(df[df["simple_journal"] == "Settled"]['amount'], hist=False)
plt.show()
tips.get_figure().savefig("kde")
# time plot
tmp = df.loc[:, ["currencycode", "creationdate", "simple_journal", "amount"]].copy()
tmp = tmp[tmp["simple_journal"] == "Chargeback"]

tmp["creationdate"] = pandas.to_datetime(tmp['creationdate'], format='%Y-%m-%d %H:%M:%S')
tmp["creationdate"] = tmp["creationdate"].dt.hour
tmp = tmp.groupby(["currencycode", "creationdate", "simple_journal"]).count().reset_index()

tips = sns.lineplot(x="creationdate", y="amount", hue="currencycode", data=tmp)
tips.set(xlabel='Hours', ylabel='count')
plt.show()
tips.get_figure().savefig("line")

# re-assign labels
df["simple_journal"] = df["simple_journal"].map({"Chargeback": 1, "Settled": 0})

# def convertCurrencyCode(x):
#     dict = {'MXN': 'MX', 'AUD': 'AU', 'NZD': 'NZ', 'GBP': 'GB', 'SEK': 'SE'}
#     currency = x["currencycode"]
#     return dict[currency]
#
# df.loc[:, "currencycode"] = df.apply(lambda x: convertCurrencyCode(x), axis=1)

# data processing
df["match"] = df["issuercountrycode"] == df["shoppercountrycode"]
df["match"] = df["match"].map({True: 1, False: 0})
df["time"] = pandas.to_datetime(df["creationdate"], format='%Y-%m-%d %H:%M:%S')
df["time"] = df["time"].dt.hour
dfp = df.loc[:, ["time", "simple_journal", "converted_amount", "match", "currencycode"]].copy()
print("columns are")
print(dfp.columns)

# solve imbalance problem
from imblearn.combine import SMOTETomek
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy import interp

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler

#ratio of positive samples among all samples
print("Ratio before smote:" + str(len(dfp[dfp["simple_journal"] == 1]) / len(dfp[dfp["simple_journal"] == 0])))

# one-hot encode
dfp = pandas.get_dummies(dfp, columns=["currencycode"], dummy_na=False)
#splite samples into label and data
data = dfp.loc[:, dfp.columns != 'simple_journal'].values
label = dfp["simple_journal"].values

smote = SMOTETomek(sampling_strategy=0.2)
datas, labels = smote.fit_resample(data, label)

rus = RandomUnderSampler(sampling_strategy=0.2)
datau, labelu = rus.fit_resample(data, label)
#reshuffle data
datas, labels = shuffle(datas, labels)
datau, labelu = shuffle(datau, labelu)
data, label = shuffle(data, label)
#Ratio after using sampling method
print("Ratio after smote:" + str(np.count_nonzero(labels) / len(labels)))
print("Ratio after undersample:" + str(np.count_nonzero(labelu) / len(labelu)))
# Four classifers
clf_1 = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf_2 = KNeighborsClassifier(n_neighbors=3)
clf_3 = svm.SVC(kernel='linear', probability=True)
clf_4 = LogisticRegression(C=500, penalty='l1', solver="liblinear")


# k-fold cross validation and drawing plots
def crossVal(X, y, num_split, clf, name, color, m):
    cv = KFold(n_splits=num_split)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for train, test in cv.split(X, y):
        if m == 2:
            rus = RandomUnderSampler(sampling_strategy=0.2)
            D, L = rus.fit_resample(X[train], y[train])
        elif m == 1:
            smote = SMOTETomek(sampling_strategy=0.2)
            D, L = smote.fit_resample(X[train], y[train])
        else:
            D = X[train]
            L = y[train]
        probas_ = clf.fit(D, L).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1], pos_label=1)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color=color,
             label=r'%s ROC (AUC = %0.2f $\pm$ %0.2f)' % (name, mean_auc, std_auc),
             lw=2, alpha=.8)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")


# plot ROC-plots
crossVal(data, label, 10, clf_1, "original", 'y', 0)
crossVal(data, label, 10, clf_1, "smote+tomek", 'g', 1)
crossVal(data, label, 10, clf_1, "undersample", 'b', 2)
plt.title('Random Forest')
plt.show()
crossVal(data, label, 10, clf_2, "original", 'y', 0)
crossVal(data, label, 10, clf_2, "smote+tomek", 'g', 1)
crossVal(data, label, 10, clf_2, "undersample", 'b', 2)
plt.title('3-NN')
plt.show()
crossVal(data, label, 10, clf_4, "original", 'y', 0)
crossVal(data, label, 10, clf_4, "smote+tomek", 'g', 1)
crossVal(data, label, 10, clf_4, "undersample", 'b', 2)
plt.title('Logistic Regression')
plt.show()
#SVM doesn't suit this assignment
# crossVal(data,label,10,clf_3,"svm")

# At least 100 fraudulent cases are found in the test data, with at most 1000 false positives.
from sklearn.metrics import confusion_matrix

#evaluate classifiers using 10-fold and optimized threshold
def evaluate(x, y, clf, num_split):
    x, y = shuffle(x, y)
    threshold = np.arange(0.1, 1.0, 0.1).tolist()
    threshold = ['%.1f' % elem for elem in threshold]
    threshold = [float(i) for i in threshold]
    cv = KFold(n_splits=num_split)
    bestF1 = 0
    bestT = 0
    bestCon = []
    aucb = 0
    for iterator in threshold:
        f_sum = 0
        tn_sum = 0
        fp_sum = 0
        fn_sum = 0
        tp_sum = 0
        auc_sum = 0
        i = 0
        for train, test in cv.split(x, y):
            #Smote
            smote = SMOTETomek(sampling_strategy=0.2)
            D, L = smote.fit_resample(x[train], y[train])
            # Undersampling
            # rus = RandomUnderSampler(sampling_strategy=0.2)
            # D, L = rus.fit_resample(data, label)
            y_prob = clf.fit(D, L).predict_proba(x[test])[:, 1]
            y_pred = y_prob > iterator
            y_pred = y_pred.astype(int)
            # if np.count_nonzero(y_pred) < 100:
            #     break
            tn, fp, fn, tp = confusion_matrix(y_pred=y_pred, y_true=y[test]).ravel()
            # if fp > 1000:
            #     break
            auc_score = roc_auc_score(y[test], y_prob)
            auc_sum += auc_score
            tn_sum += tn
            fp_sum += fp
            fn_sum += fn
            tp_sum += tp
            if tp + fp == 0:
                precision = 0
            else:
                precision = 1.0 * tp / (tp + fp)
            if tp + fn == 0:
                recall = 0
            else:
                recall = 1.0 * tp / (tp + fn)
            if precision + recall == 0:
                f1 = 0
            else:
                f1 = (2.0 * precision * recall) / (precision + recall)
            f_sum += f1
            i += 1
        f1 = f_sum / num_split
        if f1 > bestF1 and i == 10:
            bestF1 = f1
            bestT = iterator
            aucb = auc_sum / num_split
            bestCon = [tn_sum / num_split, fp_sum / num_split, fn_sum / num_split, tp_sum / num_split]
    return bestCon, bestT, bestF1, aucb

#white box KNN-3
[tn, fp, fn, tp], bestT, bestF1, aucb = evaluate(data, label, clf_2, 10)
print('-' * 10)
print('For white box - clf2')
print("precision is %0.2f   recall is %0.2f" % (tp / (tp + fp), tp / (tp + fn)))
print("average tn: %d,fp: %d, fn: %d, tp : %d" % (round(tn), round(fp), round(fn), round(tp)))
print("Accuracy is : %0.2f" % float((tp + tn) / (tp + tn + fn + fp)))
print("f1 is : %0.2f" % bestF1)
print("best threshold is %0.2f" % bestT)
print("AUC is %0.2f" % aucb)
print('-' * 10)

#Black box
[tn, fp, fn, tp], bestT, bestF1, aucb = evaluate(data, label, clf_1, 10)
print('-' * 10)
print('For black box')
print("precision is %0.2f   recall is %0.2f" % (tp / (tp + fp), tp / (tp + fn)))
print("average tn: %d,fp: %d, fn: %d, tp : %d" % (round(tn), round(fp), round(fn), round(tp)))
print("Accuracy is : %0.2f" % float((tp + tn) / (tp + tn + fn + fp)))
print("f1 is : %0.2f" % bestF1)
print("best threshold is %0.2f" % bestT)
print("AUC is %0.2f" % aucb)
print('-' * 10)

#Logistic Regression
[tn, fp, fn, tp], bestT, bestF1, aucb = evaluate(data, label, clf_4, 10)
print('-' * 10)
print('Logistic Regression')
print("precision is %0.2f   recall is %0.2f" % (tp / (tp + fp), tp / (tp + fn)))
print("average tn: %d,fp: %d, fn: %d, tp : %d" % (round(tn), round(fp), round(fn), round(tp)))
print("Accuracy is : %0.2f" % float((tp + tn) / (tp + tn + fn + fp)))
print("f1 is : %0.2f" % bestF1)
print("best threshold is %0.2f" % bestT)
print("AUC is %0.2f" % aucb)
print('-' * 10)
