import joblib


def save_object(obj, path):

    joblib.dump(
        obj,
        path
    )


def load_object(path):

    return joblib.load(path)