import pickle

def save_variable_as_pickle(variable, filename):
    """Save a Python variable to a file using pickle."""
    with open(filename, 'wb') as file:
        pickle.dump(variable, file)

def load_variable_from_pickle(filename):
    """Load a Python variable from a pickle file."""
    with open(filename, 'rb') as file:
        return pickle.load(file)

