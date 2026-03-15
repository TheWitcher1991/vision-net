# VisionNet

Комплексная система для семантической сегментации изображений, включающая REST API, веб-интерфейс и десктопное приложение.

## Структура проекта

```
vision-net/
├── backend/           # FastAPI сервис для инференса
├── frontend/          # Next.js веб-приложение
├── desktop/           # PyQt десктопное приложение
├── docs/              # Документация
└── README.MD          # Этот файл
```

## Компоненты

### Backend (FastAPI)

REST API для запуска инференса обученных моделей семантической сегментации.

- **Технологии**: FastAPI, PyTorch, Pillow
- **Порт**: 8000
- **Документация**: [docs/backend/README.md](docs/backend/README.md)

### Frontend (Next.js)

Веб-интерфейс для взаимодействия с backend API.

- **Технологии**: Next.js 16, React 19, Tailwind CSS, shadcn/ui
- **Порт**: 3000
- **Документация**: [docs/frontend/README.md](docs/frontend/README.md)

### Desktop (PyQt)

Десктопное приложение с полным функционалом: управление датасетами, обучение моделей, инференс.

- **Технологии**: PyTorch, PyQt/Tkinter, OpenCV
- **Документация**: [docs/desktop/README.md](docs/desktop/README.md)

## Быстрый старт

### Требования

- Python 3.12+
- Node.js 18+
- PyTorch 2.0+
- CUDA (опционально, для GPU)

### Запуск Backend

```bash
cd backend
poetry install
poetry run uvicorn app.main:app --reload
```

### Запуск Frontend

```bash
cd frontend
npm install
npm run dev
```

### Запуск Desktop

```bash
cd desktop
poetry install
python main.py
```

### Запуск через Docker

```bash
# Сборка и запуск всех сервисов
docker-compose up --build

# Запуск в production режиме
docker-compose -f docker-compose.prod.yml up --build
```

Для GPU версии:
```bash
docker-compose up backend-gpu --build
```

## API Endpoints

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/health` | Проверка работоспособности |
| GET | `/models` | Список доступных моделей |
| POST | `/model/load` | Загрузка модели |
| GET | `/model/status` | Статус текущей модели |
| POST | `/infer` | Запуск инференса |

## Модель

Система использует **DeepLabV3** с ResNet50 в качестве backbone для семантической сегментации.

## Лицензия

MIT