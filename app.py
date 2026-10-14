from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Покупки в интернет-магазине', layout='wide')
st.title('Какие посещения заканчиваются покупкой?')
st.write('Учебный проект: анализ завершённых посещений интернет-магазина и оценка склонности к покупке.')
if not (ROOT/'results/model.joblib').exists():
    st.error('Сначала запустите: python train.py'); st.stop()
summary = json.loads((ROOT/'results/summary.json').read_text())
metrics = pd.read_csv(ROOT/'results/metrics.csv')
model = joblib.load(ROOT/'results/model.joblib')
a,b,c = st.columns(3)
a.metric('Посещений после удаления дублей', summary['rows'])
b.metric('Доля покупок', f"{summary['purchase_rate']:.1%}")
c.metric('Выбранная модель', summary['winner'])
tabs = st.tabs(['Задача и данные', 'Анализ', 'Модели', 'Демонстрация'])
with tabs[0]:
    st.write('Вопрос: насколько модель лучше случайного выбора выделяет покупателей среди 20% посещений с самым высоким прогнозом?')
    st.markdown('[Источник: UCI](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)')
    st.write(f"Удалено дублей: {summary['duplicates_removed']}. Пропущенных ячеек: {summary['missing_cells']}.")
    st.warning('PageValues исключён из признаков из-за риска утечки. Полные характеристики посещений доступны после их завершения. Модель не доказывает эффект скидок и не является системой прогноза в реальном времени.')
with tabs[1]:
    for file, caption in [('classes','Большинство посещений не заканчивается покупкой.'), ('visitors','Сравниваем доли покупок. Это связь, а не доказательство причины.'), ('months','Месяцы имеют разный объём данных; сезонность одного года не гарантирует повторение.'), ('pages','Сравниваем число просмотренных товарных страниц.')]:
        st.image(str(ROOT/'results'/f'{file}.png'), caption=caption)
with tabs[2]:
    st.write('20% данных отложены для финального теста. Настройка параметров: 3-fold cross-validation на обучающих данных. Модель выбрана по CV PR-AUC (average precision).')
    st.dataframe(metrics.round(3), hide_index=True)
    st.write('Precision: доля покупок среди положительных прогнозов. Recall: доля найденных покупок. Lift top20: во сколько раз доля покупателей в верхних 20% выше средней.')
    st.image(str(ROOT/'results/confusion.png'))
with tabs[3]:
    demo = pd.read_csv(ROOT/'data/demo_sessions.csv')
    idx = st.selectbox('Выбери пример завершённого посещения', range(len(demo)))
    row = demo.iloc[[idx]].copy()
    row['ProductRelated'] = st.number_input('Товарных страниц', min_value=0, value=int(row.ProductRelated.iloc[0]))
    row['ProductRelated_Duration'] = st.number_input('Время на товарных страницах, секунды', min_value=0.0, value=float(row.ProductRelated_Duration.iloc[0]))
    if st.button('Рассчитать оценку'):
        score = model.predict_proba(row)[0,1]
        st.metric('Оценка модели', f'{score:.1%}')
        st.caption('Оценка не откалибрована отдельно. Изменение полей демонстрирует реакцию модели и не доказывает, что изменение поведения вызовет покупку.')
    st.dataframe(row, hide_index=True)
