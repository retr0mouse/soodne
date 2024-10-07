# Dockerfile

# Используем официальный образ Python 3.10 slim
FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Обновляем пакеты и устанавливаем необходимые зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    locales \
    && rm -rf /var/lib/apt/lists/*

# Настраиваем локаль на en_US.UTF-8
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen en_US.UTF-8

# Устанавливаем переменные окружения для локали и Python
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV PYTHONIOENCODING=utf-8
ENV PYTHONUTF8=1

# Копируем и устанавливаем Python-зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем все файлы проекта в контейнер
COPY . .

# Открываем порт 8000 для сервера Uvicorn
EXPOSE 8000

# Устанавливаем команду запуска сервера
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
