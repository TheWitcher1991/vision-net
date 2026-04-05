# Улучшения для сегментации мелких объектов

## Текущая архитектура

- **Модель**: DeepLabV3 + ResNet50 backbone
- **Input size**: 512x512
- **Классы**: 8 классов мучнистой росы + 1 background
- **Loss**: CE + Dice Loss

## Рекомендуемые улучшения

### 1. Увеличение разрешения ввода

```json
"image_size": 768  // или 1024
```

**Влияние**: Больше пикселей на мелкие структуры мучнистой росы.

---

### 2. Feature Pyramid Network (FPN)

Добавить FPN для multi-scale feature fusion:

```
model/
├── backbone (ResNet)
├── fpn (Feature Pyramid Network)
├── aspp (ASPP module)
└── decoder (upsampling + refinement)
```

**Файл**: `desktop/ml/fpn.py`

---

### 3. Attention Gates

Добавить attention gates в decoder для фокуса на мелких областях:

```python
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        # Gating signal, skip connection, intermediate channels
```

**Файл**: `desktop/ml/attention.py`

---

### 4. OhemLoss (Online Hard Example Mining)

Для сложных случаев и дисбаланса классов:

```python
class OhemLoss(nn.Module):
    def __init__(self, thresh=0.7, n_min=None):
        self.thresh = thresh
        self.n_min = n_min
```

**Файл**: `desktop/ml/loss.py`

---

### 5. Patch-based обучение

Разбиение изображений на патчи 256x256 для увеличения выборки:

```
original: 512x512 -> 4 patches x 256x256
```

**Файл**: `desktop/ml/dataset.py`

---

### 6. Cosine Annealing LR

```python
scheduler = CosineAnnealingWarmRestarts(T_0=10, T_mult=2)
```

**Файл**: `desktop/ml/trainer.py`

---

### 7. Вынос классов в конфиг

```json
// model_config.json
"classes": {
    "background": 0,
    "powdery_mildew_severity_1": 1,
    "powdery_mildew_severity_2": 2,
    ...
}
```

**Файл**: `desktop/utils/class_manager.py`

---

### 8. Data Augmentation

```python
albumentations = [
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    RandomRotate90(p=0.5),
    ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, p=0.5),
    ElasticTransform(p=0.3),
    GaussNoise(var_limit=(10, 50), p=0.3),
    GaussianBlur(blur_limit=(3, 7), p=0.3),
]
```

---

## Приоритет внедрения

| Приоритет | Улучшение | Сложность | Влияние |
|-----------|-----------|-----------|---------|
| 1 | Увеличение image_size до 768 | Low | High |
| 2 | Data Augmentation | Low | High |
| 3 | OhemLoss | Medium | High |
| 4 | Attention Gates | Medium | Medium |
| 5 | FPN | High | High |
| 6 | Patch-based training | Medium | Medium |
| 7 | Cosine Annealing | Low | Medium |

---

## Минимальный датасет

| Метрика | Минимум | Рекомендуно |
|---------|---------|-------------|
| Изображений | 500 | 5000+ |
| Разметка на изображение | 3+ объекта | 10+ объекта |
