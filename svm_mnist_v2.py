import numpy as np


def main():
    data = get_data()

    # Get classifier
    from sklearn.svm import SVC
    clf = SVC(probability=False,  # cache_size=200,
              kernel="rbf", C=2.8, gamma=.0073)

    print("Start fitting")

    # take all of it - make that number lower for experiments
    examples = len(data['train']['X'])
    clf.fit(data['train']['X'][:examples], data['train']['y'][:examples])

    analyze(clf, data)


def analyze(clf, data):
    
    # Get confusion matrix
    from sklearn import metrics
    predicted = clf.predict(data['test']['X'])
    print("Confusion matrix:\n%s" %
          metrics.confusion_matrix(data['test']['y'],
                                   predicted))
    print("Accuracy: %0.4f" % metrics.accuracy_score(data['test']['y'],
                                                     predicted))

    # Print example
    try_id = 1
    out = clf.predict(data['test']['X'][try_id])  # clf.predict_proba
    print("out: %s" % out)
    size = int(len(data['test']['X'][try_id])**(0.5))
    view_image(data['test']['X'][try_id].reshape((size, size)),
               data['test']['y'][try_id])


def get_data():
    simple = False
    if simple:  # Load the simple, but similar digits dataset
        from sklearn.datasets import load_digits
        from sklearn.utils import shuffle
        digits = load_digits()
        x = [np.array(el).flatten() for el in digits.images]
        y = digits.target

        # Scale data to [-1, 1] - This is of major importance!!!

        x = x/255.0*2 - 1

        x, y = shuffle(x, y, random_state=0)

        from sklearn.cross_validation import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.33,
                                                            random_state=42)
        data = {'train': {'X': x_train,
                          'y': y_train},
                'test': {'X': x_test,
                         'y': y_test}}
    else:  # Load the original dataset
        from sklearn.datasets import fetch_mldata
        from sklearn.utils import shuffle
        mnist = fetch_mldata('MNIST original')

        x = mnist.data
        y = mnist.target

        # Scale data to [-1, 1] - This is of mayor importance!!!
        x = x/255.0*2 - 1

        x, y = shuffle(x, y, random_state=0)

        from sklearn.cross_validation import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.33,
                                                            random_state=42)
        data = {'train': {'X': x_train,
                          'y': y_train},
                'test': {'X': x_test,
                         'y': y_test}}
    return data


if __name__ == '__main__':
    main()
