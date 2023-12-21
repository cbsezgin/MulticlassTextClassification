import pickle


def save_file(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data,f)


def load_file(filename):
    return pickle.load(open(filename, 'rb'))