# Backend API Documentation

## Обзор

FastAPI сервис для запуска инференса моделей семантической сегментации изображений.

## Требования

- Python 3.12+
- FastAPI 0.115+
- PyTorch 2.10+
- Pillow 10.0+
- Scipy 1.17+

## Установка

```bash
cd backend
poetry install
```

## Запуск

```bash
poetry run uvicorn app.main:app --reload --port 8000
```

## Структура

```
backend/
├── app/
│   ├── main.py        # FastAPI приложение
│   ├── service.py     # Логика инференса
│   └── models/        # Директория с моделями
│       └── vision_net/
│           ├── model.pth  # Веса модели
│           └── config.json # Конфигурация
├── pyproject.toml
└── start.sh
```

## API Endpoints

### GET /health

Проверка работоспособности сервиса.

**Ответ:**
```json
{
  "status": "ok"
}
```

### GET /models

Получение списка доступных моделей.

**Ответ:**
```json
{
  "models": ["vision_net"]
}
```

### POST /model/load

Загрузка модели для инференса.

**Тело запроса:**
```json
{
  "model_name": "vision_net"
}
```

**Ответ:**
```json
{
  "status": "success",
  "message": "Model 'vision_net' loaded successfully"
}
```

**Ошибки:**
- 404: Модель не найдена
- 400: Ошибка загрузки модели

### GET /model/status

Получение статуса текущей модели.

**Ответ (модель загружена):**
```json
{
  "loaded": true,
  "classes": ["background", "leaf", "fruit"],
  "device": "cpu"
}
```

**Ответ (модель не загружена):**
```json
{
  "loaded": false
}
```

### POST /infer

Запуск инференса на изображении.

**Параметры:**
- `image` (form-data): Файл изображения

**Ответ:**
```json
{
  "original_image": "base64_encoded_image",
  "mask": "base64_encoded_mask",
  "overlay": "base64_encoded_overlay",
  "detected_classes": [
    {"class": "leaf", "confidence": 0.85},
    {"class": "fruit", "confidence": 0.72}
  ],
  "class_overlays": {
    "leaf": "base64_encoded_class_overlay",
    "fruit": "base64_encoded_class_overlay"
  }
}
```

**Ошибки:**
- 400: Модель не загружена
- 500: Ошибка обработки изображения

## Использование модели

### Подготовка модели

1. Создайте директорию в `app/models/<model_name>/`
2. Сохраните веса модели в `model.pth`
3. Создайте `config.json`:

```json
{
  "num_classes": 12,
  "in_channels": 3,
  "classes": ["background", "class1", "class2", ...]
}
```

### Конфигурация InferenceService

Параметры в `service.py`:

- `image_size`: Размер входного изображения (по умолчанию: 512)
- `threshold`: Порог уверенности для сегментации (по умолчанию: 0.5)
- `device`: Устройство для инференса (cpu/cuda)

## Архитектура

### VisionNetAdapter

Адаптер модели на базе DeepLabV3 с ResNet50 backbone:

```python
class VisionNetAdapter(nn.Module):
    def __init__(self, num_classes: int = 12, pretrained: bool = False)
    def forward(self, x: torch.Tensor) -> torch.Tensor
```

### Обработка изображения

1. **Preprocess**: Изменение размера до 512x512, нормализация
2. **Predict**: Прямой проход через модель, получение вероятностей
3. **Postprocess**: Интерполяция до исходного размера, создание маски

## Примеры

### Запуск через curl

```bash
# Проверка здоровья
curl http://localhost:8000/health

# Список моделей
curl http://localhost:8000/models

# Загрузка модели
curl -X POST http://localhost:8000/model/load \
  -H "Content-Type: application/json" \
  -d '{"model_name": "vision_net"}'

# Инференс
curl -X POST http://localhost:8000/infer \
  -F "image=@image.jpg"
```

### Использование в Python

```python
import requests

# Загрузка модели
requests.post(
    "http://localhost:8000/model/load",
    json={"model_name": "vision_net"}
)

# Инференс
with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/infer",
        files={"image": f}
    )
    result = response.json()
```