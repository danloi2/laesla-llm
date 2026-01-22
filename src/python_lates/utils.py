import os


def get_model_path():
    """Returns the absolute path to the model directory."""
    return os.path.join(os.path.dirname(__file__), "model")


def do_something_useful():
    print("Replace this with a utility function")
