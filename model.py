from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import matplotlib.pyplot as plt


class Regressors(BaseEstimator, TransformerMixin):

    def __init__(self, y_test):

        from sklearn.cross_decomposition import PLSRegression
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.svm import SVR
        from xgboost import XGBRegressor

        regressor_list = [
            PLSRegression(n_components=1), SVR(),
            DecisionTreeRegressor(), RandomForestRegressor(n_estimators=1000),
            GradientBoostingRegressor(n_estimators=100),
            AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=100),
            KNeighborsRegressor(n_neighbors=5),
            AdaBoostRegressor(KNeighborsRegressor(n_neighbors=5), n_estimators=100),
            XGBRegressor()]

        self.regressors = regressor_list
        self.params = None
        self.best_model_index = None
        self.best_error = None
        self.y_test = y_test

    def fit(self, X, y):

        for index in range(len(self.regressors)):
            self.regressors[index].fit(X, y)

    def predict(self, X):

        from sklearn.metrics import mean_squared_error

        best_pred = []
        self.best_err = 999
        self.best_model_index = -1;

        for i, regressor in enumerate(self.regressors):

            y_pred = regressor.predict(X)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            if rmse < self.best_err:
                self.best_err = rmse
                best_pred = y_pred
                self.best_model_index = i

        print("The best model is: ", self.regressors[self.best_model_index],
              " with an rmse of {0:.4f}".format(self.best_err))
        print("\n")

        return best_pred


def traintest(data, targets, test_data, test_targets, onlyOffline=True):

    from sklearn.feature_selection import mutual_info_regression, SelectKBest
    from sklearn.decomposition import PCA
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.metrics import mean_squared_error

    if onlyOffline:
        print("\nOFFLINE DATA\n")

    else:
        print("\nOFFLINE + INTERPOLATED DATA\n")

    pca = PCA(n_components=20)
    univ_selector = SelectKBest(mutual_info_regression, k=30)

    feature_selector = FeatureUnion([("pca", pca), ("univ_selector", univ_selector)])

    def rmse_score(y, y_pred):

        rmse = np.sqrt(mean_squared_error(y, y_pred))

        return rmse

    cv = 5
    train_folds = np.array_split(data, cv)
    train_target_folds = np.array_split(targets, cv)
    cross_val_scores = list()

    for i in range(cv):
        new_train = list(train_folds)
        new_test = new_train.pop(i)

        target_train = list(train_target_folds)
        target_test = target_train.pop(i)
        target_train = np.concatenate(target_train)

        new_train = feature_selector.fit_transform(np.concatenate(new_train), target_train)
        new_test = feature_selector.transform(np.array(new_test))

        estimator = Regressors(y_test=target_test)
        estimator.fit(new_train, target_train)

        pred = estimator.predict(new_test)
        cross_val_scores.append(rmse_score(pred, target_test))

    print("The Cross Validation Score is: {0:.3f} +/- {0:.3f}".format(np.mean(cross_val_scores),
                                                                      np.std(cross_val_scores)))

    pca = PCA(n_components=20)
    univ_selector = SelectKBest(mutual_info_regression, k=30)
    feature_selector = FeatureUnion([("pca", pca), ("univ_selector", univ_selector)])
    estimator = Regressors(y_test=test_targets)

    pipe = Pipeline([("feature_selection", feature_selector), ("Regressor List", estimator)])
    pipe.fit(data, y=targets)
    pred = pipe.predict(test_data)
    error = rmse_score(test_targets, pred)
    print("The test error is: {0:.3f}".format(error))

    x_range = range(len(pred))
    plt.title('Predicted Values for Test Data')
    plt.plot(x_range, pred, 'r')
    plt.plot(x_range, test_targets, 'b')
    plt.legend(['Predicted Values', 'Actual Values'])
    plt.show()
