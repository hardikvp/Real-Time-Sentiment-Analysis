from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pymongo
from pymongo import MongoClient
client = pymongo.MongoClient("localhost", 27017)
db = client.test
db.my_collection

reviews = []
reviewType = []

vectorizer = CountVectorizer(min_df=0)
def make_xy(vectorizer):
    for item in db.mood_collection.find({"title" :"happy"}):
        reviews.append(item["sentence"])
        reviewType.append(1)
    for item in db.mood_collection.find({"title" :"sad"}):
        reviews.append(item["sentence"])
        reviewType.append(2)
    for item in db.mood_collection.find({"title" :"anger"}):
        reviews.append(item["sentence"])
        reviewType.append(3)

    vectorizer.fit(reviews)

    X = vectorizer.transform(reviews)
    X = X.toarray()
    y = reviewType
    y = np.array(y)

    return X, y

X, y = make_xy(vectorizer)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
clfNB = MultinomialNB()
clfSVM = svm.SVC()
clfNB.fit(Xtrain, ytrain)
clfSVM.fit(Xtrain, ytrain)

trainAccNB = clfNB.score(Xtrain, ytrain)
testAccNB = clfNB.score(Xtest, ytest)

trainAccSVM = clfSVM.score(Xtrain, ytrain)
testAccSVM = clfSVM.score(Xtest, ytest)

print ("Training Accuracy for the given set of mood statements = ", trainAccNB)
print ("Testing Accuracy for the given set of mood statements = ", testAccNB)

print ("Training Accuracy for the given set of mood statements = ", trainAccSVM)
print ("Testing Accuracy for the given set of mood statements = ", testAccSVM)


def calibration_plot(clf, Xtest, ytest):
    prob = clf.predict_proba(X)[:, 1]
    output = y
    data = pd.DataFrame(dict(prob = prob, outcome = output))

    bins = np.linspace(0, 1, 20)
    cuts = pd.cut(prob, bins)

    calibrated = data.groupby(cuts).outcome.agg(['mean', 'count'])
    calibrated['avgProbability'] = (bins[1:] + bins[:-1]) / 2
    calibrated['sigma'] = np.sqrt(calibrated.avgProbability * (1 - calibrated.avgProbability) / calibrated['count'])
    plt.errorbar(calibrated.avgProbability, calibrated['mean'], calibrated.sigma)
    plt.show()

def log_likelihood(clf, X, y):
    prob = clf.predict_log_proba(X)
    rotten = y == 0
    fresh = y == 1
    l = prob[rotten, 0].sum() + prob[fresh, 1].sum()
    return l

def cv_score(clf, X, y, score_func):
    result = 0
    nfold = 5
    kf = KFold(n_splits = nfold, shuffle = True, random_state = None)
    for train, test in kf.split(X):
        clf.fit(X[train], y[train])
        result = result + score_func(clf, X[test], y[test])
    return result / nfold

alphas = [0, .1, 1, 5]
min_dfs = [1e-3, 1e-2, 1e-1]

#Find the best value for alpha and min_df, and the best classifier
best_alpha = None
best_min_df = None
max_loglike = -np.inf

for alpha in alphas:
    for min_df in min_dfs:
        vectorizer = CountVectorizer(min_df = min_df)
        X, y = make_xy(vectorizer)
        clf = MultinomialNB(alpha = alpha)
        loglike = cv_score(clf, X, y, log_likelihood)
        if (loglike > max_loglike):
            max_loglike = loglike
            best_min_df = min_df
            best_alpha = alpha
        else:
            continue

print ("alpha: %f" % best_alpha)
print ("min_df: %f" % best_min_df)


vectorizer = CountVectorizer(min_df=best_min_df)
X, y = make_xy(vectorizer)
xtrain, xtest, ytrain, ytest = train_test_split(X, y)

clf = MultinomialNB(alpha=best_alpha).fit(xtrain, ytrain)

calibration_plot(clf, xtest, ytest)

# Your code here. Print the accuracy on the test and training dataset
training_accuracy = clf.score(xtrain, ytrain)
test_accuracy = clf.score(xtest, ytest)

print ("Accuracy on training data: %0.2f" % (training_accuracy))
print ("Accuracy on test data:     %0.2f" % (test_accuracy))


words = np.array(vectorizer.get_feature_names())

X = np.eye(Xtest.shape[1])
probs = clf.predict_log_proba(X)[:, 0]
ind = np.argsort(probs)

good_words = words[ind[:10]]
bad_words = words[ind[-10:]]

good_prob = probs[ind[:10]]
bad_prob = probs[ind[-10:]]

print ("Good words\t     P(fresh | word)")
for w, p in zip(good_words, good_prob):
    print ("%20s" % w, "%0.2f" % (1 - np.exp(p)))

print ("Bad words\t     P(fresh | word)")
for w, p in zip(bad_words, bad_prob):
    print ("%20s" % w, "%0.2f" % (1 - np.exp(p)))

testReview = '''Dear Members of the Carnegie Mellon University Community,

Welcome to the new academic year!

CMU stands at a pivotal moment in its history, positioned like no other institution to define and manage the space where human life and technology meet, to address complex problems and to inspire creative expression. We offer a distinctive, world-class education, in which a new generation is cultivating deep disciplinary knowledge, while learning collaboration and leadership. Our momentum is fueled by the incredible energy of our global community, united by a common purpose, pride and spirit.

Part of CMU’s promise lies in promoting a diverse and inclusive community, a university-wide commitment that has led to a very exciting milestone. For the first time in our history, more than half of our 1,670 incoming undergraduates on the Pittsburgh campus are women. More than ever, across every school and college, the best students in the world are choosing CMU.

In welcoming our students, including 2,981 exceptional new graduate students, we also are reminded of how much we all gain from such a supportive network of classmates, colleagues, mentors and advisers. We are never alone. Lean on the people around you when you need help, and freely offer your talents, passions and support to others.

Whether you are a returning or incoming student, a member of our faculty or of our staff, I encourage you to seize the opportunities this new year offers. Claim your role in propelling this great university to new heights, even as you find your own path to success. I’m thrilled to be a part of this remarkable community at this important moment, and I look forward to sharing the journey with you.

Wishing you a successful year, and looking forward to seeing you around campus,'''

tr = vectorizer.transform([testReview])
print (clfNB.predict_proba(tr))
