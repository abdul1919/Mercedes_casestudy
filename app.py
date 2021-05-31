from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import time
import joblib
from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin
from sklearn.utils import check_array
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor

#########################################################################################

train_data = pd.read_csv('train.csv')
Y_train = train_data['y']

#########################################################################################

import flask
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/index', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        path = "C:\\Users\\abdul\\Downloads\\TRAIL\\uploads\\" + file.filename
        file.save(path)
        test_data = pd.read_csv(path)
        test_df = test_data.copy()
        train_data = pd.read_csv('train.csv')
        
        for c in test_df.columns:
            if test_df[c].dtype == 'object':
                mapper = lambda x:sum([ord(digit) for digit in x])
                test_df[c] = test_data[c].apply(mapper)
                train_data[c] = train_data[c].apply(mapper)
                
        X_test = test_df
        
        class Stacking(BaseEstimator,TransformerMixin):
            def __init__(self, estimator):
                self.estimator = estimator

            def fit(self, X, y=None, **fit_params):
                self.estimator.fit(X, y, **fit_params)
                return self
            def transform(self, X):
                X = check_array(X) # Converting into array format
                X_new = np.copy(X) # Taking copy of existing X data
         # Checking for classification model and if so, checking also the model has attribute 'predict_proba'
                if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
                    X_new = np.hstack((self.estimator.predict_proba(X), X)) # Stacking predicted values along with dataset

                X_new = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_new)) # Stacking predicted values along with dataset
                return X_new
            
        stacked_pipeline = make_pipeline(
        Stacking(estimator=LassoLarsCV(normalize=True)),
        Stacking(estimator=GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=3, max_features=0.55, min_samples_leaf=18,         min_samples_split=14, subsample=0.7)),
        LassoLarsCV())
        
        X_train = train_data.drop(['y'],axis=1)
        Y_train = train_data['y']
        
        stacked_pipeline.fit(X_train,Y_train)
        pred_normal = stacked_pipeline.predict(X_test)

        # Decomposed features

        tsvd = joblib.load('tvsd.pkl')
        tsvd_results_test = tsvd.transform(test_df)
        pca = joblib.load('pca.pkl')
        pca2_results_test = pca.transform(test_df)
        ica = joblib.load('ica.pkl')
        ica2_results_test = ica.transform(test_df)
        grp = joblib.load('grp.pkl')
        grp_results_test = grp.transform(test_df)
        srp = joblib.load('srp.pkl')
        srp_results_test = srp.transform(test_df)

        # Including all the decomposed features into test dataset
        n_comp = 12
        for i in range(n_comp):
    
            # Including pca features
            test_df['pca_' + str(i)] = pca2_results_test[:, i]

            # Including ica features
            test_df['ica_' + str(i)] = ica2_results_test[:, i]

            # Including svd features
            test_df['tsvd_' + str(i)] = tsvd_results_test[:, i]

            # Including grp features
            test_df['grp_' + str(i)] = grp_results_test[:, i]

            # Including srp features
            test_df['srp_' + str(i)] = srp_results_test[:, i]

        X_test_decom = test_df

        gbr = joblib.load('gbr.pkl')
        pred_decom = gbr.predict(X_test_decom)
        final = pred_decom*0.75 + pred_normal*0.25 # This will give scalar value

        test_data['test_time'] = final
        return test_data.to_html(header="true",  table_id="table" )
    return  flask.render_template('index.html', message=' Upload file')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)