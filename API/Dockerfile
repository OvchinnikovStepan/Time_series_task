# Берем легкую версию Python
FROM python:3.12-slim

# Устанавливаем рабочую папку внутри контейнера
WORKDIR /app

# Копируем список зависимостей и ставим их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем наши файлы в контейнер
COPY  /app .

# Запускаем основной файл
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]