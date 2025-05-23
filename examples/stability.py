import optuna

from labelfree.data.load import load_shuttle, load_fraud

if __name__ == "__main__":

    x_train, x_test, y_test = load_fraud(setup=True)

    def objective(trial):

        


