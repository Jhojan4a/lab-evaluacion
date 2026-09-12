from src.data_loader import generate_and_save_data, load_data
from src.preprocessing import prepare_data
from src.supervised_model import train_and_eval_classifier
from src.unsupervised_model import run_unsupervised
from sklearn.preprocessing import StandardScaler

def main():
    print("Iniciando Pipeline de Evaluacion...")
    generate_and_save_data(n_samples=1000)
    X, y = load_data()
    
    X_train_s, X_test_s, y_train, y_test, _ = prepare_data(X, y)
    train_and_eval_classifier(X_train_s, X_test_s, y_train, y_test)
    
    scaler_global = StandardScaler()
    X_scaled = scaler_global.fit_transform(X)
    run_unsupervised(X_scaled)
    print("\n[+] Fin de la ejecucion exitosa.")

if __name__ == "__main__":
    main()