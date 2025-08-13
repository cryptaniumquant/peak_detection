# Trading Signal Bot

Автоматизированный бот для обнаружения торговых сигналов с использованием пиковой детекции и уведомлений через Telegram.

## 🚀 Быстрый старт

### 🐳 Docker (Рекомендуется для продакшена)

**Самый простой способ запуска на любой системе:**

```bash
# 1. Создайте local_settings.py с вашими настройками
# 2. Запустите одной командой
chmod +x run.sh
./run.sh run
```

**Управление контейнером:**
```bash
./run.sh logs     # Просмотр логов
./run.sh status   # Статус контейнера
./run.sh restart  # Перезапуск
./run.sh shell    # Доступ к shell контейнера
./run.sh stop     # Остановка
./run.sh clean    # Полная очистка
```

### 💻 Локальная установка (для разработки)

```bash
# Клонируйте репозиторий и перейдите в директорию проекта
cd peak_detection_dev

# Запустите автоматическую установку
python setup.py
```

Скрипт установки автоматически:
- Создаст виртуальную среду в `bot_service/.venv`
- Установит все зависимости
- Проверит конфигурацию
- Протестирует импорты

## ⚙️ Конфигурация

### 1. Создание local_settings.py

Создайте файл `local_settings.py` в корневой директории:

```python
# Database credentials
DB_HOST = 'your_host'
DB_PORT = 'your_port'
DB_DATABASE = 'your_database'
DB_USER = 'your_user'
DB_PASSWORD = 'your_password'

# Telegram bot credentials
TELEGRAM_BOT_TOKEN = 'your_bot_token'
TELEGRAM_CHAT_ID = 'your_chat_id'

# Bot settings
MODE = 'real'  # or 'simulate'
REAL_DETECT_HOURS = 25
VIZ_WINDOW_DAYS = 7
TIMEZONE = 'Europe/Moscow'
```

### 2. Настройка пороговых значений

Отредактируйте `strategy_thresholds.json`:

```json
{
  "strategy_name_1": -18.0,
  "strategy_name_2": -25.0
}
```

### 3. Локальный запуск

```bash
# Просто запустите из корневой директории
python run_bot.py
```

## 📁 Структура проекта

```
peak_detection_dev/
├── run_bot.py              # 🚀 Главный скрипт запуска
├── setup.py                # 🔧 Скрипт установки
├── config.py               # ⚙️ Основная конфигурация
├── local_settings.py       # 🔐 Локальные настройки (создать)
├── strategy_thresholds.json # 📊 Пороговые значения стратегий
├── calculate_peak.py       # 📈 Алгоритм пиковой детекции
├── strategy_data_processor.py # 🗄️ Обработка данных БД
├── Dockerfile              # 🐳 Docker образ
├── entrypoint.sh           # 🔧 Docker entrypoint
├── run.sh                  # 🚀 Docker управление
├── .dockerignore           # 🚫 Docker исключения
├── bot_service/            # 🤖 Сервис бота
│   ├── .venv/             # 🐍 Виртуальная среда
│   ├── run_bot.py         # 🤖 Основной код бота
│   ├── requirements.txt   # 📦 Зависимости
│   ├── services/          # 🛠️ Сервисы бота
│   │   ├── data_pipeline.py
│   │   ├── notifier.py
│   │   ├── state_store.py
│   │   └── visualization_service.py
│   └── state/             # 💾 Состояние бота
└── README.md              # 📖 Этот файл
```

## 🔧 Команды бота

- `/run_now` - Запустить цикл обнаружения сигналов вручную
- `/simulate_at <strategy> <YYYY-MM-DD HH:MM>` - Симуляция сигнала в конкретное время
- `/all_viz` - Создать визуализации для всех стратегий

## 🐳 Docker Deployment

### Системные требования
- Docker и Docker Compose
- Linux/macOS/Windows с WSL2
- Минимум 1GB RAM
- Доступ к MySQL базе данных

### Особенности Docker версии
- **Автоматический перезапуск** при сбоях
- **Ротация логов** (100MB, 3 файла)
- **Изоляция зависимостей** от хост-системы
- **Правильная настройка timezone** (Europe/Moscow)
- **Безопасность**: credentials монтируются отдельно

### Переменные окружения

- `MODE` - Режим работы: `real` или `simulate`
- `REAL_DETECT_HOURS` - Часы для обнаружения в реальном режиме (по умолчанию: 25)
- `VIZ_WINDOW_DAYS` - Дни для визуализации (по умолчанию: 7)
- `TIMEZONE` - Часовой пояс (по умолчанию: Europe/Moscow)

### Расписание

Бот автоматически запускается каждый час в начале часа (XX:00) используя APScheduler с cron-триггером.

## 🗄️ База данных

Система использует **асинхронные** запросы к базе данных с помощью:
- SQLAlchemy async engine
- asyncmy драйвер для MySQL
- Оптимизированные запросы для лучшей производительности

## 🔍 Алгоритм обнаружения

1. **Сбор данных**: Получение последних N часов данных PnL
2. **Сглаживание**: Применение фильтра Савицкого-Голая (окно=25, полином=1)
3. **Пиковая детекция**: Анализ первой и второй производных
4. **Пороговые значения**: Использование абсолютных порогов или динамических квантилей
5. **Cooldown система**: 24ч блокировка + 24ч восстановление
6. **Уведомления**: Отправка сигналов через Telegram с графиками

## 🐛 Отладка и мониторинг

### Docker логи
```bash
./run.sh logs        # Просмотр логов
./run.sh shell       # Доступ к контейнеру
docker logs trading-bot -f  # Следить за логами в реальном времени
```

### Локальная отладка
```bash
python -c "from config import load_settings; print('Config OK')"
```

### Проверка состояния
```bash
./run.sh status      # Статус контейнера
docker ps -f name=trading-bot  # Детальная информация
```

## 📊 Мониторинг

Бот логирует:
- Успешные подключения к БД
- Обнаруженные сигналы и их параметры
- Ошибки обработки данных
- Статистику по стратегиям
- Отправленные уведомления
- Ошибки выполнения и их причины

## 🚨 Troubleshooting

### Частые проблемы

**Docker контейнер не запускается:**
```bash
# Проверьте логи
./run.sh logs

# Проверьте конфигурацию
./run.sh shell
cat local_settings.py
```

**Нет подключения к БД:**
- Проверьте параметры подключения в `local_settings.py`
- Убедитесь, что БД доступна из Docker контейнера
- Проверьте firewall настройки

**Telegram бот не отвечает:**
- Проверьте `TELEGRAM_BOT_TOKEN` и `TELEGRAM_CHAT_ID`
- Убедитесь, что бот добавлен в чат
- Проверьте интернет соединение контейнера

**Нет сигналов:**
- Проверьте пороговые значения в `strategy_thresholds.json`
- Убедитесь, что в БД есть свежие данные
- Проверьте режим работы (`MODE` в настройках)

### Полезные команды

```bash
# Перезапуск с полной пересборкой
./run.sh clean && ./run.sh run

# Мониторинг ресурсов
docker stats trading-bot

# Экспорт логов
docker logs trading-bot > bot_logs.txt 2>&1
```

## 🔄 Обновления

Для обновления зависимостей:
```bash
cd bot_service
.venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt --upgrade
```

## 📝 Логи

Все логи выводятся в консоль. Для сохранения в файл:
```bash
python run_bot.py > bot.log 2>&1
```

---

**💡 Совет**: Используйте `python setup.py` для первоначальной настройки и `python run_bot.py` для запуска бота.
