import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

#Load the data
x_train = joblib.load("data/processed/x_train.pkl")
y_train = joblib.load("data/processed/y_train.pkl")
x_test  = joblib.load("data/processed/x_test.pkl")
y_test  = joblib.load("data/processed/y_test.pkl")

#Train eExtreme Gradient Boosting(xgboost) model
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    scale_pos_weight=1,
    eval_metric='logloss'
)

xgb.fit(x_train, y_train)

#Evaluate
preds = xgb.predict(x_test)
print(classification_report(y_test, preds))

#Save xgboost model
#( The produced model .pkl file will be at root folder, 
# so move the model to this /idsmodel directory )
joblib.dump(xgb, "baseline_xgb.pkl")