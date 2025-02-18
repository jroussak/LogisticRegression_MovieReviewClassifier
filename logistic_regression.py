# CS115B Spring 2025 Homework 1
# Logistic Regression Classifier
import random
from collections import defaultdict, Counter
import os
import numpy as np
import scipy

class LogisticRegression():

    def __init__(self):
        self.class_dict = {}
        # use of self.feature_dict is optional for this assignment
        self.feature_dict = {}
        self.n_features = None
        self.theta = None  # weights (and bias)
        # Personal additions
        self.neg_words = {"disappointing", "predictable", "worst", "meaningless", "offensive",
    "terrible", "useless", "disappointment", "racist", "stupid",
    "boring", "sexist", "waste", "bland", "amateur",
    "dumb", "run-of-the-mill", "dull", "flawed", "bad",
    "weak", "worse", "fail", "failed", "failure",
    "awful", "insulting", "lifeless", "cliche", "poor",
    "uninspired", "lame", "tasteless", "fails", "wasted",
    "senseless", "repetitive", "disappointed", "empty", "lacking", "lacks", "lacked", "frivolous", "random",
    "devoid", "cheesy", "hideous", "hideously", "flop", "flopped", "flops", "flopping"}

        self.pos_words = {"touching", "funniest", "awesome", "genius", "stunning",
    "unique", "super", "perfect", "beautiful", "refreshing",
    "wonderful", "amazing", "better", "interesting", "love",
    "loved", "sweet", "spectacular", "cool", "best",
    "enjoyed", "incredible", "enjoy", "favorite", "great",
    "funny", "hilarious", "fantastic", "excellent", "emotional",
    "wonderfully", "rare", "special", "spectacularly", "fantastically", "amazingly", "imaginative", "chemistry",
    "blast", "creative", "entertaining", "warm"}
#     "active", "adaptable", "adventurous", "affectionate", "alert",
#     "artistic", "assertive", "boundless", "brave", "broad-minded",
#     "calm", "capable", "careful", "caring", "cheerful",
#     "clever", "comfortable", "communicative", "compassionate", "conscientious",
#     "considerate", "courageous", "creative", "curious", "decisive",
#     "determined", "diligent", "dynamic", "eager", "energetic",
#     "entertaining", "enthusiastic", "exuberant", "expressive", "fabulous",
#     "fair-minded", "fantastic", "fearless", "flexible thinker", "frank",
#     "friendly", "funny", "generous", "gentle", "gregarious",
#     "happy", "hard working", "helpful", "hilarious", "honest",
#     "imaginative", "independent", "intellectual", "intelligent", "intuitive",
#     "inventive", "joyous", "kind", "kind-hearted", "knowledgeable",
#     "level-headed", "lively", "loving", "loyal", "mature",
#     "modest", "optimistic", "outgoing", "passionate", "patient",
#     "persistent", "philosophical", "polite", "practical", "pro-active",
#     "productive", "quick-witted", "quiet", "rational", "receptive",
#     "reflective", "reliable", "resourceful", "responsible", "selective",
#     "self-confident", "sensible", "sensitive", "skillful", "straightforward",
#     "successful", "thoughtful", "trustworthy", "understanding", "versatile",
#     "vivacious", "warm-hearted", "willing", "witty", "wonderful"
# }
        self.punc = {".", "?", "...", "!"}
        self.vocab = dict()


    '''
    Given a training set, fills in self.class_dict (and optionally,
    self.feature_dict), as in HW1.
    Also sets the number of features self.n_features and initializes the
    parameter vector self.theta.
    '''
    def make_dicts(self, train_set):
        with os.scandir(train_set) as training_set_folder:
            for i, class_folder in enumerate(training_set_folder):
                if class_folder.name == ".DS_Store": continue # hidden (?) file contained in the data sets
                self.class_dict[class_folder.name] = i - 1
                with os.scandir(class_folder) as c_f:
                    for file in c_f:
                        with open(file) as f:
                            for line in f:
                                for word in line.strip().split():
                                    self.vocab[word] = len(self.vocab)
        self.n_features = len(self.vocab) + 2
        self.theta = np.array([0 for _ in range(self.n_features)] + [1])
        # print(f"vocab: \n {self.vocab}")


    '''
    Loads a dataset. Specifically, returns a list of filenames, and dictionaries
    of classes and documents such that:
    classes[filename] = class of the document
    documents[filename] = feature vector for the document (use self.featurize)
    '''
    def load_data(self, data_set):
        filenames = []
        classes = {}
        documents = {}
        with os.scandir(data_set) as d_s:
            for class_folder in d_s:
                if class_folder.name == ".DS_Store": continue
                # lexicon = set([word for word, count in self.get_lexicon(class_folder.path).most_common(1000)])
                # if self.class_dict[class_folder.name] == 1:
                #     self.pos_words = lexicon
                #     # print(f"pos lexicon: {self.pos_words}")
                # else:
                #     self.neg_words = lexicon
                #     # print(f"neg lexicon: {self.neg_words}")
                with os.scandir(class_folder) as c_f:
                    for file in c_f:
                        file_name = os.path.basename(file)
                        with open(file) as f:
                            filenames.append(file_name)
                            classes[file_name] = self.class_dict[class_folder.name]
                            documents[file_name] = self.featurize(f)
        return filenames, classes, documents


    """
    Given a specific class_folder, e.g. neg, create a Counter lexicon with all of the words in all of its documents
    """
    def get_lexicon(self, class_folder):
        counts = Counter()
        with os.scandir(class_folder) as c_f:
            for file in c_f:
                with open(file) as f:
                    for line in f:
                        counts.update(line.split())
        return counts


    '''
    Given a document (as a list of words), returns a feature vector.
    Note that the last element of the vector, corresponding to the bias, is a
    "dummy feature" with value 1.
    '''
    def featurize(self, document):
        vector = [0 for _ in range(self.n_features)] + [0]
        for line in document:
            for word in line.split(" "):
                # if word == " " or word == "\n": continue
                # feature 0: positive words
                if word in self.pos_words:
                    vector[0] += 1
                # feature 1: negative words
                elif word in self.neg_words:
                    vector[1] += 1
                if word in self.vocab:
                    vector[2 + self.vocab[word]] += 1 # first 2 vectors are standard, the latter ~60k are counts
                # elif word == '!':
                #         vector[2] += 1
            # feature 3: # of lines in doc
            # vector[3] += 1
        # print(f"vector = {[p for p in vector if p != 0]}")
        return vector


    '''
    Trains a logistic regression classifier on a training set.
    '''
    def train(self, train_set, batch_size=3, n_epochs=1, eta=0.1):
        for epoch in range(n_epochs):
            filenames, classes, documents = self.load_data(train_set)
            random.Random(epoch).shuffle(filenames)
            data_covered = 0
            cross_entropy_loss = np.array([0.0 for _ in range(batch_size)])
            while data_covered < len(filenames) - batch_size:
                cur_batch_size = min(batch_size, len(filenames) - data_covered)
                data_covered += cur_batch_size
                # 1. Create vectors
                x = np.array([documents[f] for f in filenames[data_covered:data_covered + cur_batch_size]])
                y = np.array([classes[f] for f in filenames[data_covered:data_covered + cur_batch_size]])
                # 2. Compute y_hat
                y_hat = scipy.special.expit(np.dot(x, self.theta))
                # 3. Update CE Loss
                for i in range(len(y_hat)):
                    cross_entropy_loss[i] += -(y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat))[i]
                # 4. Calculate avg gradient over mini_batch
                avg_gradient = (1 / cur_batch_size) * (np.matmul(np.transpose(x), (y_hat - y)))
                # 5. Update weights with learning rate and gradient
                self.theta = self.theta - (eta * avg_gradient)


    '''
    Tests the classifier on a development or test set.
    Returns a dictionary of filenames mapped to their correct and predicted
    classes such that:
    results[filename]['correct'] = correct class
    results[filename]['predicted'] = predicted class
    '''
    def test(self, dev_set):
        results = defaultdict(dict)
        filenames, classes, documents = self.load_data(dev_set)
        for filename in filenames:
            results[filename]['correct'] = classes[filename]
            results[filename]['predicted'] = (1 if
            scipy.special.expit(np.matmul(documents[filename], self.theta)) >= 0.5 else 0)
        return results


    '''
    Given results, calculates the following:
    Precision, Recall, F1 for each class
    Accuracy overall
    Also, prints evaluation metrics in readable format.
    '''
    def evaluate(self, results):
        true_pos = Counter()
        false_pos = Counter()
        false_neg = Counter()
        total_correct = 0
        for f in results:
            correct = results[f]['correct']
            predicted = results[f]['predicted']
            if correct == predicted:
                true_pos[correct] += 1
                total_correct += 1
            else:
                false_pos[predicted] += 1
                false_neg[correct] += 1

        for class_name in self.class_dict:
            class_val = self.class_dict[class_name]
            prec_denom = (true_pos[class_val] + false_pos[class_val])
            if prec_denom == 0:
                prec_denom = float('inf')
            prec = true_pos[class_val] / prec_denom
            rec_denom = (true_pos[class_val] + false_neg[class_val])
            if rec_denom == 0:
                rec_denom = float('inf')
            rec = true_pos[class_val] / rec_denom
            f1_denom = (prec + rec)
            if f1_denom == 0:
                f1_denom = float('inf')
            f1 = 2 * prec * rec / f1_denom
            print(f"Class: {class_name}")
            print(f"Precision: {prec}")
            print(f"Recall:    {rec}")
            print(f"F1:        {f1}")
        acc = total_correct / len(results)
        print(f"Overall accuracy:  {acc}")
        print(f"--------------------------------------")
        return acc


if __name__ == '__main__':
    best_acc = 0
    best_params = [0, 0, 0]
    i = 1
    batch_sizes = [1]
    num_epochs = [14, 15, 16, 17, 18]#[64, 32, 16]
    etas = [.15, .16, .17, .18]#[.08]#, .15, .18, .2]#[0.01, 0.2, 0.4]
    num_runs = len(batch_sizes) * len(num_epochs) * len(etas)
    for b_size in batch_sizes:
        for num_epoch in num_epochs:
            for e in etas:
                LR = LogisticRegression()
                LR.make_dicts(train_set="movie_reviews/train")
                print(f"({i} / {num_runs}) Initiating LogReg with batch size {b_size}, epochs: {num_epoch}, "
                      f"eta: {e}")
                LR.train("movie_reviews/train",
                         batch_size=b_size, n_epochs=num_epoch, eta=e)
                results = LR.test("movie_reviews/dev")
                acc = LR.evaluate(results)
                i += 1
                if acc >= best_acc:
                    best_acc = acc
                    best_params = [b_size, num_epoch, e]
    print(f"best acc was {best_acc} with {best_params}")
    print(f"-----------------------------------------------------")
