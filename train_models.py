import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             RocCurveDisplay)
import joblib

# Загрузка данных
df = pd.read_csv('travel_data.csv')
X = df.drop('recommend', axis=1)
y = df['recommend']

# Разделение
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Предобработка
numeric_features = ['rating', 'budget']
categorical_features = ['season']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Модели
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}
best_model = None
best_f1 = 0

# Для графиков
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
conf_matrices = {}
roc_curves = {}

for idx, (name, model) in enumerate(models.items()):
    print(f"\n--- Обучение {name} ---")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    
    # Обучаем и замеряем время (опционально)
    import time
    start = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"Время обучения: {elapsed:.2f} сек")
    
    # Предсказания
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
    
    # Метрики
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    
    results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1, 'ROC-AUC': roc}
    print(f"F1 = {f1:.4f}, ROC-AUC = {roc:.4f}")
    
    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    conf_matrices[name] = cm
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
    axes[idx].set_title(f'{name}\nConfusion Matrix')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')
    
    # ROC-кривая (если есть predict_proba)
    if y_proba is not None:
        RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=axes[idx] if False else None, name=name)
    
    if f1 > best_f1:
        best_f1 = f1
        best_model = pipeline

plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.show()

# График сравнения метрик
metrics_df = pd.DataFrame(results).T
metrics_df.plot(kind='bar', figsize=(10, 6), colormap='viridis')
plt.title('Сравнение моделей по метрикам')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('metrics_comparison.png')
plt.show()

# Feature Importance (для моделей, которые его поддерживают)
for name, model in models.items():
    if hasattr(model, 'feature_importances_'):
        # Обучаем конвейер снова (если хотим важность) – можно использовать лучшую модель
        pipeline_temp = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        pipeline_temp.fit(X_train, y_train)
        # Получаем имена признаков после препроцессинга
        preprocessor.fit(X_train)
        feature_names = (numeric_features + 
                         [f"season_{cat}" for cat in preprocessor.named_transformers_['cat'].categories_[0][1:]])
        importances = pipeline_temp.named_steps['classifier'].feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(8, 4))
        plt.title(f"Feature Importance - {name}")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig(f'feature_importance_{name.replace(" ", "_")}.png')
        plt.show()

# Кривые обучения для лучшей модели (например, Random Forest)
def plot_learning_curve(estimator, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10), 
        scoring='f1'
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(8, 5))
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="orange")
    plt.plot(train_sizes, train_mean, 'o-', color="blue", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="orange", label="Cross-validation score")
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("F1 score")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(f'learning_curve_{title.replace(" ", "_")}.png')
    plt.show()

# Обучаем лучшую модель на всех данных для кривой обучения (или можно использовать уже обученную)
plot_learning_curve(best_model, X_train, y_train, f'Learning Curve - Best Model (F1={best_f1:.3f})')

# Сохраняем лучшую модель
joblib.dump(best_model, 'best_model.pkl')
print(f"\n✅ Лучшая модель сохранена в best_model.pkl с F1 = {best_f1:.4f}")

# Печать итоговой таблицы
print("\n📊 Итоговое сравнение моделей:")
print(metrics_df.round(4))