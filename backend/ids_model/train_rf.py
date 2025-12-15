import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

#Load the data
x_train = joblib.load("data/processed/x_train.pkl")
y_train = joblib.load("data/processed/y_train.pkl")
x_test  = joblib.load("data/processed/x_test.pkl")
y_test  = joblib.load("data/processed/y_test.pkl")

#Train Random forest(RF) model
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf.fit(x_train, y_train)

#Evaluate
preds = rf.predict(x_test)
print(classification_report(y_test, preds))

#Save RF model
#(The produced model .pkl file will be at root folder, so move the model to this /idsmodel directory)
joblib.dump(rf, "baseline_rf.pkl") 

