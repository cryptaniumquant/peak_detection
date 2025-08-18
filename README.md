# Trading Signal Bot

Автоматизированный бот для обнаружения торговых сигналов с использованием алгоритма пиковой детекции Savitzky-Golay и уведомлений через Telegram.

## ✨ Ключевые особенности

- 🔍 **Алгоритм пиковой детекции** - Savitzky-Golay фильтр для точного определения сигналов
- 📱 **Telegram интеграция** - Автоматические уведомления с графиками
- 🔐 **Безопасность** - Авторизация команд по chat_id
- 📊 **Визуализация** - Автоматическое создание графиков для анализа
- 🐳 **Docker готовность** - Полная контейнеризация для продакшена
- 📝 **Структурированное логирование** - Консоль + Telegram уведомления об ошибках
- ⏰ **Планировщик** - Автоматический запуск по расписанию
- 💾 **Персистентное состояние** - Сохранение состояния между перезапусками

## 🚀 Быстрый старт

### 🐳 Docker (Рекомендуется)

```bash
# 1. Создайте local_settings.py с вашими настройками
# 2. Запустите одной командой
chmod +x run.sh
./run.sh run
```

**Управление контейнером:**
```bash
./run.sh build    # Сборка образа
./run.sh run      # Запуск контейнера
./run.sh logs     # Просмотр логов
./run.sh status   # Статус контейнера
./run.sh restart  # Перезапуск
./run.sh shell    # Доступ к shell контейнера
./run.sh stop     # Остановка
./run.sh clean    # Полная очистка
```

### 💻 Локальная разработка

```bash
# Клонируйте репозиторий
git clone <repository-url>
cd peak_detection_dev

# Автоматическая установка
python setup.py

# Или ручная установка
python -m venv bot_service/.venv
source bot_service/.venv/bin/activate  # Linux/Mac
# bot_service\.venv\Scripts\activate   # Windows
pip install -r bot_service/requirements.txt

# Запуск бота
python bot_service/run_bot.py
```

## ⚙️ Конфигурация

### 1. Создание local_settings.py

Создайте файл `local_settings.py` в корневой директории:

```python
# Обязательные настройки
TELEGRAM_BOT_TOKEN = 'your_bot_token_here'
TELEGRAM_CHAT_ID = 123456789  # ID чата для уведомлений

# База данных (async SQLAlchemy URI)
SQLALCHEMY_DATABASE_URI = 'mysql+asyncmy://user:password@host:port/database'

# Опциональные настройки
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR
MODE = 'real'  # 'real' или 'simulation'
STRATEGY_WHITELIST = []  # Пустой список = все стратегии из CSV

# Настройки планировщика (только расписание)
SCHEDULER_JOB_CONFIG = {
    'trigger': 'cron',
    'minute': 0,  # Запуск каждый час в начале часа
    'hour': '*',  # Каждый час
}
```

### 2. Настройка пороговых значений

Отредактируйте `strategy_thresholds.json` для настройки индивидуальных порогов:

```json
{
  "strategy_name_1": -18.0,
  "strategy_name_2": -25.0
}
```

## 📁 Структура проекта

```
peak_detection_dev/
├── bot_service/            # 🤖 Основной модуль бота
│   ├── run_bot.py         # 🚀 Главный скрипт запуска
│   ├── services/          # 📦 Сервисы бота
│   │   ├── data_pipeline.py    # 🔄 Пайплайн обработки данных
│   │   ├── notifier.py         # 📱 Telegram уведомления
│   │   ├── state_store.py      # 💾 Управление состоянием
│   │   └── visualization_service.py # 📊 Создание графиков
│   └── requirements.txt   # 📋 Python зависимости
├── config.py              # ⚙️ Единая конфигурация
├── logging_config.py      # 📝 Настройка логирования
├── local_settings.py      # 🔐 Локальные настройки (создать)
├── calculate_peak.py      # 📈 Алгоритм пиковой детекции
├── strategy_data_processor.py # 🗄️ Обработка данных БД
├── strategy_thresholds.json # 📊 Пороговые значения стратегий
├── Dockerfile             # 🐳 Docker конфигурация
├── run.sh                 # 🔧 Скрипт управления Docker
└── setup.py               # 🛠️ Автоматическая установка
```

## 🤖 Telegram команды

После запуска бота доступны следующие команды:

- `/run_now` - Запустить анализ немедленно
- `/simulate_at YYYY-MM-DD HH:MM` - Симуляция на определенное время
- `/all_viz` - Создать графики для всех стратегий

**Важно:** Команды работают только для авторизованного chat_id из конфигурации.

## 📊 Логирование и мониторинг

Система логирования включает:

- **Консольные логи** - Все события выводятся в консоль
- **Telegram алерты** - Критические ошибки автоматически отправляются в чат
- **Уровни логирования** - DEBUG, INFO, WARNING, ERROR
- **Ротация логов** - Автоматическая ротация в Docker

Настройка уровня логирования в `local_settings.py`:
```python
LOG_LEVEL = 'INFO'  # DEBUG для детальной отладки
```

## 🐳 Docker особенности

### Персистентные данные

Контейнер автоматически создает volume-маппинги для:

- `./visualizations` → `/root/app/visualizations` - Временные графики
- `./bot_service/state` → `/root/app/bot_service/state` - Состояние бота
- `./local_settings.py` → `/root/app/local_settings.py` - Конфигурация (read-only)

### Автоматический перезапуск

Контейнер настроен с `--restart unless-stopped` для автоматического восстановления после сбоев.

## 🔧 Разработка

### Локальная отладка
```bash
python -c "import config; print('Config OK')"
python -c "from bot_service.services import data_pipeline; print('Pipeline OK')"
```

### Проверка состояния
```bash
./run.sh status
./run.sh logs
```

## 📈 Алгоритм

Бот использует алгоритм Savitzky-Golay для сглаживания временных рядов и детекции пиков:

1. **Загрузка данных** - Получение данных стратегий из БД
2. **Ресэмплинг** - Агрегация к часовым данным
3. **Фильтрация** - Применение Savitzky-Golay фильтра
4. **Детекция пиков** - Поиск точек ребалансировки
5. **Уведомления** - Отправка сигналов в Telegram с графиками

## 🚨 Troubleshooting

### Частые проблемы

**Ошибка импорта config:**
```bash
# Убедитесь что запускаете из правильной директории
cd peak_detection_dev
python -m bot_service.run_bot
```

**Нет подключения к БД:**
- Проверьте `SQLALCHEMY_DATABASE_URI` в `local_settings.py`
- Убедитесь что используете async драйвер (`mysql+asyncmy://`)

**Telegram бот не отвечает:**
- Проверьте `TELEGRAM_BOT_TOKEN` и `TELEGRAM_CHAT_ID`
- Убедитесь что бот добавлен в чат

### Логи Docker
```bash
./run.sh logs -f  # Следить за логами в реальном времени
docker logs peak_detection_bot --tail 100
```

## 📝 Changelog

### v2.0.0 (Текущая версия)
- ✅ Полное рефакторинг логирования
- ✅ Telegram уведомления об ошибках  
- ✅ Авторизация команд по chat_id
- ✅ Упрощенная архитектура конфигурации
- ✅ Оптимизированный Docker setup
- ✅ Персистентное состояние
- ✅ Структурированные логи

---

**Автор:** Cryptanium Quant Team  
**Лицензия:** MIT  
**Поддержка:** [GitHub Issues](https://github.com/cryptaniumquant/peak_detection/issues)
