import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC, SVC
from scipy.sparse import csr_matrix

def load_data(data_path):
    def sparse_to_dense(sparse_r_d, vocab_size):
        r_d = [0.0 for _ in range(vocab_size)]
        indices_tfidfs = sparse_r_d.split()
        for index_tfidf in indices_tfidfs:
            index = int(index_tfidf.split(':')[0])
            tfidf = float(index_tfidf.split(':')[1])
            r_d[index] = tfidf
        return np.array(r_d)

    with open(data_path) as f:
        d_lines = f.read().splitlines()
    with open('session 1/data/20news-bydate/words_idfs.txt') as f:
        vocab_size = len(f.read().splitlines())

    data = []
    labels = []
    for data_id, d in enumerate(d_lines):
        features = d.split('<fff>')
        label, doc_id = int(features[0]), int(features[1])
        r_d = sparse_to_dense(features[2], vocab_size)
        labels.append(label)
        data.append(r_d)
        
    return data, labels

def clustering_with_KMeans():
    data, labels = load_data('session 1/data/20news-bydate/tfidf_full.txt')
    X = csr_matrix(data)
    print('=' * 10)
    kmeans = KMeans(
        n_clusters=20,
        init='random',
        n_init=5,
        tol=1e-3,
        random_state=2018
    ).fit(X)
    labels = kmeans.labels_
    
    print(labels)

def compute_accuracy(predicted_y, expected_y):
    matches = np.equal(predicted_y, expected_y)
    accuracy = np.sum(matches.astype(float)) / len(expected_y)
    return accuracy

def classifying_with_SVMs():
    train_X, train_y = load_data('session 1/data/20news-bydate/tfidf_train.txt')
    train_X, train_y = np.array(train_X), np.array(train_y)
    classifier = LinearSVC(
        C=10.0, # penalty coefficient
        tol=1e-3, # tolerance for stopping criteria
        verbose=False # whether prints out logs or not
    )
    classifier.fit(train_X, train_y)
    
    test_X, test_y = load_data('session 1/data/20news-bydate/tfidf_test.txt')
    test_X, test_y = np.array(test_X), np.array(test_y)
    predicted_y = classifier.predict(test_X)
    accuracy = compute_accuracy(predicted_y, test_y)
    with open('session 2/result.txt', 'a') as f:
        f.write(f'Linear SVM:\n')
        f.write(f'Accuracy: {accuracy}\n')
    
    classifier2 = SVC(
        C=50.0,
        kernel='rbf',
        gamma=0.1,
        tol=1e-3,
        verbose=True
    )
    classifier2.fit(train_X, train_y)
    
    predicted_y = classifier2.predict(test_X)
    accuracy = compute_accuracy(predicted_y, test_y)
    with open('session 2/result.txt', 'a') as f:
        f.write(f'Kernel SVM:\n')
        f.write(f'Accuracy: {accuracy}\n')


if __name__ == '__main__':
    classifying_with_SVMs()