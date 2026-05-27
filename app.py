import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Загрузка модели
model = joblib.load('best_model.pkl')

st.set_page_config(page_title="Travel Recommendation", layout="centered")
st.title("🧳 Рекомендация мест для путешествий")
st.markdown("Введите параметры поездки, и модель скажет — стоит ли рекомендовать это место.")

# Поля ввода
rating = st.slider("⭐ Средняя оценка места (1-5)", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
budget = st.number_input("💰 Бюджет поездки (USD)", min_value=50, max_value=10000, value=1000, step=50)
season = st.selectbox("🌤️ Сезон", options=['winter', 'spring', 'summer', 'autumn'])

# Кнопка прогноза
if st.button("Узнать рекомендацию"):
    # Создаём DataFrame с правильными именами столбцов
    input_df = pd.DataFrame([[rating, budget, season]], 
                            columns=['rating', 'budget', 'season'])
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    
    if prediction == 1:
        st.success(f"✅ Рекомендую! (вероятность: {proba:.2%})")
    else:
        st.error(f"❌ Не рекомендую. (вероятность положительной рекомендации: {proba:.2%})")
    
    st.info("Модель учитывает: высокий рейтинг, соответствие бюджета сезону и отсутствие экстремально низких/высоких бюджетов.")