# Evaluación de Laboratorio: Inteligencia Artificial

**Estudiante:** Jhojan Abel Yana Ramos  
**Dominio:** Telemetría de Servidores de Red y Diagnóstico de Fallas en Data Center  
**Dataset:** 1000 instancias sintéticas | 4 variables continuas | 1 variable objetivo binaria  

---

## 1. Arquitectura Modular del Proyecto
El pipeline se diseñó bajo la metodología Open Code desacoplando responsabilidades funcionales:

* `data/dataset.csv`: Dataset sintético de 1000 registros con métricas de hardware (`cpu_usage_pct`, `ram_usage_gb`, `network_throughput_mbps`, `temperature_celsius`, `server_status`).
* `src/data_loader.py`: Generación con distribuciones controladas mediante `make_classification` y carga desacoplada.
* `src/preprocessing.py`: Partición estratificada Train/Test (80/20) y ajuste de `StandardScaler` exclusivamente sobre el conjunto de entrenamiento para evitar fuga de información (*data leakage*).
* `src/supervised_model.py`: Entrenamiento y evaluación de un clasificador de Regresión Logística.
* `src/unsupervised_model.py`: Reducción dimensional con PCA a 2 componentes principales, clustering con K-Means ($k=3$), cómputo del Coeficiente de Silueta y exportación de la gráfica 2D.
* `main.py`: Script orquestador de todo el pipeline.

---

## 2. Resultados Experimentales

### 2.1. Pipeline Supervisado (Regresión Logística)
* **Partición de datos:** Train (800 instancias) - Test (200 instancias)
* **Accuracy Global:** 92.00% (0.9200)
* **F1-Score Ponderado:** 0.9200
* **Matriz de Confusión:**
  ```text
  [[112    8]
   [  8   72]]