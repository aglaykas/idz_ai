import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 10000

# Признаки
ratings = np.random.uniform(2.5, 5.0, n_samples)  # средние оценки мест
budgets = np.random.randint(100, 5000, n_samples)  # бюджет поездки в USD
seasons = np.random.choice(['winter', 'spring', 'summer', 'autumn'], n_samples)

# Целевая переменная (логическое правило)
def generate_recommendation(rating, budget, season):
    # Высокий рейтинг (>4.0) всегда рекомендуется
    if rating > 4.0:
        return 1
    # Для сезонов: летом предпочтительнее бюджет 500-3000, зимой 300-2000 и т.д.
    season_rules = {
        'winter': (300, 2000),
        'spring': (400, 2500),
        'summer': (500, 3000),
        'autumn': (400, 2200)
    }
    min_b, max_b = season_rules[season]
    # Если бюджет в подходящем диапазоне и рейтинг >=3.0
    if min_b <= budget <= max_b and rating >= 3.0:
        return 1
    # Если бюджет слишком низкий (<300) или слишком высокий (>4000) - не рекомендуется
    if budget < 300 or budget > 4000:
        return 0
    # Иначе случайный шум (10% случайных ошибок)
    return np.random.choice([0, 1], p=[0.1, 0.9]) if rating > 3.5 else 0

recommendations = [generate_recommendation(r, b, s) for r, b, s in zip(ratings, budgets, seasons)]

# Формирование DataFrame
df = pd.DataFrame({
    'rating': np.round(ratings, 1),
    'budget': budgets,
    'season': seasons,
    'recommend': recommendations
})

# Сохраняем
df.to_csv('travel_data.csv', index=False)
print(f"Датасет создан: {len(df)} записей")
print(df.head())