from sklearn.ensemble import VotingClassifier

def build_ensemble(rf_model, xgb_model):
    """
    Soft voting ensemble for robust IDS
    """
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('xgb', xgb_model)
        ],
        voting='soft'
    )

    return ensemble
