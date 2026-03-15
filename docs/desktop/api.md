# API Documentation

## ML Modules

### ml.model

#### VisionNetAdapter

Основная модель семантической сегментации на базе DeepLabV3.

```python
from ml.models import VisionNetAdapter, create_model, load_model

# Создание модели
model = create_model(num_classes=12)

# Загрузка модели
model, config = load_model("model.pth", device="cuda")
```

**Конструктор:**
```python
VisionNetAdapter(num_classes: int = 12, pretrained: bool = True, **kwargs)
```

| Параметр | Тип | Описание |
|----------|-----|----------|
| num_classes | int | Количество классов для сегментации |
| pretrained | bool | Использовать предобученные веса |
| kwargs | dict | Дополнительные аргументы для DeepLabV3 |

**Методы:**
- `forward(x: torch.Tensor) -> torch.Tensor` - прямой проход

---

### ml.trainer

#### Trainer

Класс для обучения модели.

```python
from ml.trainer import Trainer

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device="cuda",
    learning_rate=0.001,
    on_log=log_callback,
    on_epoch_end=epoch_callback
)

trainer.train(epochs=10)
trainer.stop()
```

**Конструктор:**
```python
Trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    device: str = "cpu",
    learning_rate: float = 0.001,
    on_log: callable = None,
    on_epoch_end: callable = None
)
```

| Параметр | Тип | Описание |
|----------|-----|----------|
| model | nn.Module | Модель для обучения |
| train_loader | DataLoader | Загрузчик обучающих данных |
| val_loader | DataLoader | Загрузчик валидационных данных |
| device | str | Устройство (cpu/cuda) |
| learning_rate | float | Скорость обучения |
| on_log | callable | Callback для логирования |
| on_epoch_end | callable | Callback после каждой эпохи |

**Методы:**
- `train(epochs: int)` - запуск обучения в отдельном потоке
- `validate() -> float` - валидация модели
- `stop()` - остановка обучения

---

### ml.inference

#### Inference

Класс для инференса (распознавания изображений).

```python
from ml.inference import Inference

inference = Inference(
    model=model,
    device="cuda",
    classes=["background", "leaf", "fruit"],
    image_size=512,
    threshold=0.5
)

result = inference.infer("image.jpg")
# result = {
#     "original_image": PIL.Image,
#     "mask": PIL.Image,
#     "detected_classes": [("leaf", 0.85)],
#     "probabilities": torch.Tensor
# }
```

**Конструктор:**
```python
Inference(
    model: nn.Module,
    device: str = "cpu",
    classes: list = None,
    image_size: int = 512,
    threshold: float = 0.5
)
```

| Параметр | Тип | Описание |
|----------|-----|----------|
| model | nn.Module | Обученная модель |
| device | str | Устройство для инференса |
| classes | list | Список имен классов |
| image_size | int | Размер входного изображения |
| threshold | float | Порог уверенности |

**Методы:**
- `infer(image_path: str) -> dict` - запуск инференса на изображении
- `preprocess(image_path)` - предобработка изображения
- `predict(image_tensor)` - предсказание модели
- `postprocess(probabilities, original_size)` - постобработка результата

---

### ml.dataset

#### COCOSegmentationDataset

Датасет для семантической сегментации в формате COCO.

```python
from ml.dataset import COCOSegmentationDataset

dataset = COCOSegmentationDataset(
    root="path/to/dataset",
    annotations_file="annotations.json",
    image_size=512
)

image, mask = dataset[0]
classes = dataset.get_classes()
stats = dataset.get_stats()
```

**Конструктор:**
```python
COCOSegmentationDataset(
    root: str,
    annotations_file: str,
    image_size: int = 512,
    transform: callable = None
)
```

**Методы:**
- `__len__() -> int` - количество изображений
- `__getitem__(idx) -> (image, mask)` - получение элемента
- `get_classes() -> list` - список классов
- `get_stats() -> dict` - статистика датасета

---

### ml.loss

#### DiceLoss

Функция потерь Dice для семантической сегментации.

```python
from ml.loss import DiceLoss, VisionNetLoss

dice_loss = DiceLoss()
total_loss = VisionNetLoss(num_classes=12, device="cuda")
```

---

## Utils Modules

### utils.config_manager

#### ConfigManager

Управление конфигурацией модели.

```python
from utils.config_manager import ConfigManager

# Создание конфига
config = ConfigManager.create_model_config(
    model_name="VisionNetAdapter",
    in_channels=3,
    num_classes=12,
    image_size=512,
    classes=["class1", "class2", ...]
)

# Сохранение
ConfigManager.save_model_config(config, "config.json")

# Загрузка
config = ConfigManager.load_model_config("config.json")
```

---

### utils.coco_parser

#### COCOParser

Парсер файлов аннотаций COCO.

```python
from utils.coco_parser import COCOParser

parser = COCOParser("annotations.json")
parser.load()

images = parser.images
annotations = parser.annotations
categories = parser.categories

stats = parser.get_stats()
names = parser.get_category_names()
```

---

### utils.image_utils

#### Функции для работы с изображениями

```python
from utils.image_utils import (
    create_overlay,
    generate_colors,
    normalize_image,
    denormalize_image,
    image_to_bytes,
    bytes_to_image
)

# Создание наложения маски на изображение
overlay = create_overlay(original, mask, alpha=0.5)

# Генерация цветов для классов
colors = generate_colors(num_classes=10)
```

---

## GUI Modules

### gui.main_window

#### MainWindow

Главное окно приложения.

```python
from gui.main_window import MainWindow

app = MainWindow()
app.run()
```

### gui.dataset_panel

#### DatasetPanel

Панель управления датасетом (наследует `ttk.Frame`).

### gui.training_panel

#### TrainingPanel

Панель обучения модели (наследует `ttk.Frame`).

### gui.inference_panel

#### InferencePanel

Панель инференса (наследует `ttk.Frame`).

### gui.logs_panel

#### LogsPanel

Панель отображения логов (наследует `ttk.Frame`).

```python
# Добавление сообщения в логи
logs_panel.log("[INFO] Training started")
```
