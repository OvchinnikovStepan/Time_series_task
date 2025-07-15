# Time_series_dashboard

## Описание

**Time_series_dashboard** - проект по созданию информационной панели для анализа временных рядов и прогнозов


### Технологии
- **Язык:** Python  
- **Фреймворк:** Streamlit  
- **Библиотеки:** Pandas, Statsmodels, NumPy, Prophet  
- **База данных:** Не используется (операции выполняются в памяти)

---

## Запуск проекта

**Порт:** `8501`  
**URL:** `http://localhost:8501`

Команда запуска:
```bash
streamlit run main.py --server.port 8501
``` 

---

## Docker
Для создания образа, в корневой папке прописать:
```bash
docker build -f Dockerfile_streamlit -t streamlit-dashboard .
```
Для запуска, там же прописать:
```bash
docker run -p8501:8501 streamlit-dashboard  
````

---

## Подключение к API

Для подключения к дашборду необходимо узнать ip машины на которой запущен контейнер и  записать его в виде:

```bash
api_url = "http://ваш_ip:8000/api/"
```

В файл secrets.toml, внутри папки .streamlit в корневой папке проекта


# TimeSeriesPredictor API

## Описание

**TimeSeriesPredictor** - REST API для прогнозирования временных рядов на основе исторических данных.  
Сервис позволяет загружать данные, задавать параметры модели и получать прогнозные значения.

### Технологии
- **Язык:** Python  
- **Фреймворк:** FastAPI  
- **Библиотеки:** Pandas, Statsmodels, NumPy, Prophet  
- **База данных:** Не используется (операции выполняются в памяти)

---

## Запуск проекта

**Порт:** `8000`  
**URL:** `http://localhost:8000`

Команда запуска:
```bash
uvicorn API.app.main:app --host 0.0.0.0 --port 8000 --reload
``` 
---
## Docker
Для создания образа, в папке API прописать:
```bash
docker build -f Dockerfile_API -t model-app:1.0 . 
```
Для запуска, там же прописать:
```bash
docker run -p8000:8000 model-app:1.0  
``` 

# Документация

Общая документация к дашборду и API находится по [ссылке](https://docs.google.com/document/d/1gjq6F0gUAaUqlRZSmQ-j4LpmUwC0Ga1mhht8jNpGQsM/ "Переход к google документу с документацией")
