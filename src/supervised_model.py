from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def train_and_eval_classifier(X_train, X_test, y_train, y_test):
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n--- [SUPERVISADO: LOGISTIC REGRESSION] ---")
    print(f"Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    print(f"F1-Score: {f1:.4f}")
    print("Matriz de Confusion:\n", cm)
    return model, acc, f1, cm