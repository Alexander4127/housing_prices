# Отчёт по Домашнему заданию 2, ML System Design


**Студент:** Куцаков Александр Сергеевич

**Кейс:** Предсказания стоимости дома по табличным признакам

**Тип данных:** Табличные

**Бизнес-цель:** Регрессия

---


### **1. Эксперименты с моделями**

**Примечание:** Для удобства финальную метрику привожу в процентах (умножаю на 100).

| Модель | Основная метрика | Оптимизация гиперпараметров |
| --- | --- | --- |
| Ridge | 4.43 | Нет |
| SVM | 16.30 | Нет |
| Tree | 7.93 | Нет |
| RandomForest | 4.22 | Нет |
| RandomForest | 4.21 | GridSearch |
| RandomForest | 4.11 | Optuna |
| CatBoost | 4.36 | Нет |
| CatBoost | 4.11 | GridSearch |
| CatBoost | 3.99 | Optuna |

**Выбранная итоговая модель:** CatBoostRegressor

**Причина выбора:** Работает лучше всего по качеству, эмпирически быстрее случайного леса (который единственный сравним по качеству). Минус - требует оптимизации гиперпараметров, но в нашем случае это не проблема.

---

### **2. Демо инференса**

✔ Тип интерфейса: Streamlit

✔ Как запустить: `streamlit run infer.py`

✔ Воспроизводимость: `pip install -r reqs.txt`

---

📎 **Скриншоты метрик и интерфейса приложить отдельными файлами**

**Примечание:** Все скриншоты прикреплены в `exp.ipynb` в конце нотбука.
