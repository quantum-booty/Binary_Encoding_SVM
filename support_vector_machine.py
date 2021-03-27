import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn import datasets
import pandas as pd

plt.style.use('seaborn-darkgrid')


class SVM:
    def __init__(self):
        """
        Binary hard margin classifier!
        Probably wouldn't work for non-linearly separable case.
        """
        self.W = None
        self.W0 = None
        self.lambda_ = None
        self.margin = None

    def fit(self, X, Y):
        """
        X: [n_samples, n_features]
        Y: [n_samples]
        Y assumes the classes are +1 and -1
        """
        self.maximise_lagrangian(X, Y)
        self.compute_W_W0(Y, X)
        self.compute_training_margin()

    def maximise_lagrangian(self, X, Y):
        """Finds the support vectors and their lambdas"""

        def negative_lagrangian_objective_function(lambda_, X, Y):
            """We would like to maximise lagrangian, so when using a minimizer,
            we need to minimise the negative of the lagrangian."""
            M = lambda_[:, None] * Y * X
            return -(np.sum(lambda_) - 1 / 2 * np.sum(M @ M.T))

        N = len(X)

        # define the constraints
        # constraint lambda >= 0
        bounds = ((0, None),) * N
        # constraints sum(lambda_i * y_i) = 0
        constraints = {'type': 'eq', 'fun': lambda lambda_: np.dot(lambda_, Y)}

        # randomly intialise lambda (taking lambda >= 0 into account)
        lambda_init = np.random.randint(0, np.max(X), size=N)

        # maximise the lagrangian to find the lambdas
        res = minimize(
            negative_lagrangian_objective_function, lambda_init, args=(X, Y), bounds=bounds, constraints=constraints
        )

        # round it to get rid of noise, and so that only support vectors have non-zero lambda
        self.lambda_ = res.x.round(10)

    def compute_W_W0(self, Y, X):
        self.W = np.sum(self.lambda_[:, None] * Y * X, axis=0)

        # support vectors satisfies this condition
        # using column vector convention, numpy uses row vector convention
        # y_i * (W.T @ x_i + W_0) = 1
        # W_0 = - (y_i * W.T @ x_i / y_i)
        # We calculate W_0 for each sample i, and take its average value
        support_vec_idx = self.lambda_ != 0  # first find the support vectors
        y = Y[support_vec_idx]
        x = X[support_vec_idx]
        numerator = y * (x @ self.W)[:, None] - 1
        denominator = -y
        self.W0 = np.mean(numerator / denominator)

    def compute_training_margin(self):
        self.margin = 2 / np.linalg.norm(self.W)

    def predict(self, X):
        # The decision boundary split the data into margin > 0 and margin < 0
        # we define those > 0 as class 1 and < 0 as class -1
        preds = (X @ self.W + self.W0) > 0
        return (preds == 0) * -1 + (preds == 1)

    @staticmethod
    def feature_mapping(X, func):
        return np.apply_along_axis(func, axis=1, arr=X)

    def plot_decision_boundary(self, X, Y, X_test=None, save_figure=False, figure_path='binary_encoding_SVM_1.png'):
        if np.abs(self.W[1]) < 0.0001:
            x_range = np.array([-self.W0, -self.W0])
            y_range = np.array([np.min(X[:, 1]), np.max(X[:, 1])])
        else:
            x_range = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 2)
            y_range = -(self.W0 + x_range * self.W[0]) / self.W[1]

        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c=Y)
        if X_test is not None:
            plt.scatter(X_test[:, 0], X_test[:, 1], c='g', label='Test set')

        if self.lambda_ is not None:
            support_vectors = X[self.lambda_ != 0]
            y_support = Y[self.lambda_ != 0].ravel()
            plt.scatter(support_vectors[y_support == -1, 0], support_vectors[y_support == -1, 1], c='red')
            plt.scatter(
                support_vectors[y_support == 1, 0], support_vectors[y_support == 1, 1], c='red', label='Support vectors'
            )
            plt.legend()
        plt.plot(x_range, y_range)
        if save_figure:
            plt.savefig(figure_path, dpi=300)
        plt.show()


class BinaryEncoderSVM:
    """
    Data -> encoder -> SVMs predictions -> decoder -> y_pred
    """

    def __init__(self):
        self.decodings = None
        self.SVMs: list[SVM] = []
        self.n_svm = None

    def fit(self, X, Y, plot=False):
        """
        X: [n_samples, n_features]
        Y: [n_samples]
        Assumes y ranges from 0 to n_class - 1
        """
        classes = np.unique(Y)
        self.n_svm = int(np.ceil(np.log(len(classes)) / np.log(2)))

        self.decoding_table(classes)

        for i, y_encoded in enumerate(self.binary_encoding(Y)):
            y_encoded = y_encoded[:, None]

            svm = SVM()
            svm.fit(X, y_encoded)
            self.SVMs.append(svm)

            if plot:
                svm.plot_decision_boundary(
                    X, y_encoded, save_figure=True, figure_path=f'outputs/binary_encoding_SVM_{i}'
                )

    def binary_encoding(self, Y):
        """Encode the target labels for training the SVMs
        This is done by converting the decoding table into a table like this
        for the 3 classes case:
            Encoding      | +1  | -1
            -----------------------
            SVM 1 classes | 0 1 |  2
            SVM 2 classes | 0 2 |  1

        For example, when training SVM 1, we encode the target label such that
        class 0 and 1 has target label -1 and class 2 has target label +1.
        """
        # convert decoding table to encodings
        assert self.decodings is not None
        assert self.n_svm is not None

        for svm_idx in range(self.n_svm):
            y_encoded = np.empty_like(Y)
            for cls, code in enumerate(self.decodings):
                # change the labels according to the decoding table
                y_encoded[Y == cls] = code[svm_idx]
            yield y_encoded

    def decoding_table(self, classes):
        """
        The decoding table function produces an array called decodings, which is a table like this in array form:
            class | SVM 1 | SVM 2
            ---------------------
            0     |   1   |   1
            1     |   1   |  -1
            2     |  -1   |   1

        This table is used for decoding the predictions from all the SVMs.
        For example, if SVM 1 predicts 1 and SVM 2 predicts -1, then we would predict 1 for that sample
        """
        # convert to class number to binary, assumes the classes range from 0 to n_class - 1
        decodings = ['{0:b}'.format(c) for c in classes]
        # used for padding the decodings so the numbers can be inverted
        max_code_len = np.max([len(code) for code in decodings])
        # pad the codes with 0s so they all have same length
        decodings = [((max_code_len - len(code)) * '0' + code) for code in decodings]
        # get the final 1 and -1 decodings
        decodings = [[int(val.replace('1', '-1').replace('0', '1')) for val in code] for code in decodings]
        # print(decodings)
        self.decodings = np.array(decodings)

    def predict(self, X):
        """Predict the actual classes"""
        n_samples = len(X)
        assert self.n_svm is not None
        svm_pred_all = np.empty((n_samples, self.n_svm))
        for svm_idx, svm in enumerate(self.SVMs):
            svm_pred = svm.predict(X)
            svm_pred_all[:, svm_idx] = svm_pred

        y_pred = self.decoder_prediction(svm_pred_all)
        return y_pred

    def decoder_prediction(self, svm_preds):
        """
        Helper method for decoding the predictions from the SVMs to make the actual class predictions
        svm_preds are outputs from the SVMs of shape [n_sample, n_svm]
        """
        y_preds = []
        for pred in svm_preds:
            y_pred = np.argwhere(np.all(pred == self.decodings, axis=1))
            # if we cannot find the svm prediction in the decoding table, it means
            # that it cannot be unambiguously classified
            if y_pred.size == 0:
                y_pred = None
            else:
                y_pred = y_pred.item()
            y_preds.append(y_pred)

        return np.array(y_preds)

    @staticmethod
    def score(Y_pred, Y):
        return (Y_pred == Y).mean()


if __name__ == "__main__":

    X, y = datasets.make_blobs(
        n_samples=10 * 3,
        centers=3,
        n_features=2,
        center_box=(0, 10),
        cluster_std=1.2,
        random_state=20116366,
    )

    plt.figure()
    plt.plot(X[y == 0, 0], X[y == 0, 1], 'g^', label='class 0')
    plt.plot(X[y == 1, 0], X[y == 1, 1], 'bs', label='class 1')
    plt.plot(X[y == 2, 0], X[y == 2, 1], 'rx', label='class 2')
    plt.legend()

    clf = BinaryEncoderSVM()
    clf.fit(X, y, plot=True)
    accuracy = clf.score(clf.predict(X), y)
    print('accuracy of binary encoder svm: ', accuracy)

    X_test = (X[y == 0] + X[y == 1] + X[y == 0]) / 3
    svm_preds = np.empty((len(X_test), 2))
    for i, (y_encoded, svm) in enumerate(zip(clf.binary_encoding(y), clf.SVMs)):
        svm.plot_decision_boundary(X, y_encoded, X_test, save_figure=True, figure_path=f'outputs/testset_{i}.png')
        svm_preds[:, i] = svm.predict(X_test)

    y_test_pred = clf.predict(X_test)

    result = pd.DataFrame(
        index=range(len(X_test)),
        columns=[
            'Test sample feature 1',
            'Test sample feature 2',
            'Output of SVM 1',
            'Output of SVM 2',
            'Classification',
        ],
    )
    result['Test sample feature 1'] = X_test[:, 0].round(3)
    result['Test sample feature 2'] = X_test[:, 1].round(3)
    result['Output of SVM 1'] = svm_preds[:, 0].astype(int)
    result['Output of SVM 2'] = svm_preds[:, 1].astype(int)
    result['Classification'] = y_test_pred

    print(result)
    result.to_csv('outputs/test_set_results.csv')
