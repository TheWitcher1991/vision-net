# Архитектура VisionNet Studio

## Обзор

VisionNet Studio - это десктопное приложение для семантической сегментации изображений, построенное на архитектуре MVC с использованием PyTorch для ML и Tkinter для GUI.

```
┌─────────────────────────────────────────────────────────────┐
│                     MainWindow (Tk)                         │
├─────────────┬─────────────┬──────────────┬──────────────────┤
│ DatasetPanel│TrainingPane│InferencePanel│    LogsPanel     │
└─────────────┴─────────────┴──────────────┴──────────────────┘
```

## Слои приложения

### GUI Layer (`gui/`)

| Файл | Назначение |
|------|------------|
| `main_window.py` | Главное окно, управление вкладками |
| `dataset_panel.py` | Загрузка и отображение информации о датасете |
| `training_panel.py` | Настройка и запуск обучения |
| `inference_panel.py` | Загрузка модели и распознавание изображений |
| `logs_panel.py` | Отображение логов |

### ML Layer (`ml/`)

#### Model (`models/vision_net.py`)

```
VisionNetAdapter
├── Backbone: DeepLabV3 + ResNet50
├── Classifier: Заменяемый слой для N классов
└── Auxiliary Classifier: Для промежуточных предсказаний
```

Основные классы:
- `VisionNetAdapter` - адаптер DeepLabV3 для пользовательских классов
- `create_model()` - фабрика для создания модели
- `load_model()` - загрузка сохраненной модели

#### Trainer (`trainer.py`)

```
Trainer
├── Device Management: CPU/CUDA
├── Loss Functions: CrossEntropy + Dice
├── Optimizer: Adam с ReduceLROnPlateau
├── Mixed Precision: torch.amp.GradScaler
└── Callbacks: on_log, on_epoch_end
```

#### Inference (`inference.py`)

```
Inference
├── Preprocess: Resize, normalize, tensor conversion
├── Predict: Model inference с softmax
└── Postprocess: Resize mask, class detection
```

#### Dataset (`dataset.py`)

```
COCOSegmentationDataset
├── COCO JSON parsing
├── Image/mask pairing
├── Polygon to mask conversion
└── Data augmentation (transform)
```

### Utils Layer (`utils/`)

| Файл | Назначение |
|------|------------|
| `config_manager.py` | Управление конфигурацией модели |
| `coco_parser.py` | Парсинг COCO аннотаций |
| `image_utils.py` | Утилиты для работы с изображениями |

## Поток данных

### Обучение

```
DatasetPanel → COCOSegmentationDataset → DataLoader
                                           ↓
                                      Trainer
                                           ↓
                                    VisionNetAdapter
                                           ↓
                                      save_model()
```

### Инференс

```
InferencePanel → load_model() → Inference.infer()
                                           ↓
                                   PIL Image
                                           ↓
                                  [original, mask, overlay]
```

## Конфигурация модели

```json
{
    "model_name": "VisionNetAdapter",
    "in_channels": 3,
    "num_classes": 9,
    "image_size": 512,
    "classes": ["stem", "flower", "fruit", ...]
}
```

## Зависимости модулей

```
main.py
  └─ gui/main_window.py
       ├─ gui/dataset_panel.py
       │    └─ utils/coco_parser.py
       ├─ gui/training_panel.py
       │    ├─ ml/dataset.py
       │    ├─ ml/trainer.py
       │    ├─ ml/models/vision_net.py
       │    └─ utils/config_manager.py
       ├─ gui/inference_panel.py
       │    ├─ ml/inference.py
       │    ├─ ml/models/vision_net.py
       │    └─ utils/image_utils.py
       └─ gui/logs_panel.py
```

## GPU Управление

Автоматическое определение CUDA:
- Проверка `torch.cuda.is_available()`
- Выбор GPU по умолчанию если доступен
- Mixed precision (FP16) для ускорения обучения
- Очистка памяти между эпохами
