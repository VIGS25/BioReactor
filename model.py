from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Regressors(BaseEstimator, TransformerMixin):

    def __init__(self, y_test, logger):

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
        self.logger = logger

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

        self.logger.info("The best model is: {}".format(self.regressors[self.best_model_index]))
        self.logger.info("The model has an rmse of {0:.4f}".format(self.best_err))
        self.logger.info("\n")

        return best_pred


def traintest(data, targets, test_data, test_targets, logger, pdf_name, var='Titer', onlyOffline=True):

    from sklearn.feature_selection import mutual_info_regression, SelectKBest
    from sklearn.decomposition import PCA
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.metrics import mean_squared_error

    if onlyOffline:
        logger.info("\nOFFLINE DATA\n")

    else:
        logger.info("\nOFFLINE + INTERPOLATED DATA\n")

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
        logger.info("\n Cross Validation Fold No. {}".format(i + 1))

        new_train = list(train_folds)
        new_test = new_train.pop(i)

        target_train = list(train_target_folds)
        target_test = target_train.pop(i)
        target_train = np.concatenate(target_train)

        new_train = feature_selector.fit_transform(np.concatenate(new_train), target_train)
        new_test = feature_selector.transform(np.array(new_test))

        estimator = Regressors(y_test=target_test, logger=logger)
        estimator.fit(new_train, target_train)

        pred = estimator.predict(new_test)
        cross_val_scores.append(rmse_score(pred, target_test))

    pca = PCA(n_components=20)
    univ_selector = SelectKBest(mutual_info_regression, k=30)
    feature_selector = FeatureUnion([("pca", pca), ("univ_selector", univ_selector)])
    estimator = Regressors(y_test=test_targets, logger=logger)

    pipe = Pipeline([("feature_selection", feature_selector), ("Regressor List", estimator)])
    pipe.fit(data, y=targets)

    logger.info("\n Testing")
    pred = pipe.predict(test_data)
    error = rmse_score(test_targets, pred)

    x_range = range(len(pred))
    plt.title('Predicted Values for Test Data for variable {}'.format(var))
    plt.plot(x_range, pred, 'r')
    plt.plot(x_range, test_targets, 'b')
    plt.legend(['Predicted Values', 'Actual Values'])
    plt.savefig(pdf_name, format='pdf')

    def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                         header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                         bbox=[0, 0, 1, 1], header_columns=0,
                         ax=None, **kwargs):
        if ax is None:
            size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
            fig, ax = plt.subplots(figsize=size)
            ax.axis('off')

        mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(font_size)

        for k, cell in six.iteritems(mpl_table._cells):
            cell.set_edgecolor(edge_color)
            if k[0] == 0 or k[1] < header_columns:
                cell.set_text_props(weight='bold', color='w')
                cell.set_facecolor(header_color)
            else:
                cell.set_facecolor(row_colors[k[0] % len(row_colors)])
        return ax

    import six
    df = pd.DataFrame()
    df['Variable'] = [var + '      ']
    df['RMSE-CV'] = ["{0:.3f} +/- {0:.3f}".format(np.mean(cross_val_scores), np.std(cross_val_scores))]
    df['RMSE-Test'] = ["{0:.3f}      ".format(error)]

    ax = render_mpl_table(df, header_columns=0, col_width=3.0)
    plt.savefig(pdf_name, format='pdf')

