# Frontend Documentation

## Обзор

Веб-интерфейс для взаимодействия с Backend API. Позволяет выбирать модели, загружать изображения и просматривать результаты семантической сегментации.

## Требования

- Node.js 18+
- Next.js 16+
- React 19+

## Установка

```bash
cd frontend
npm install
```

## Запуск

```bash
npm run dev
```

Приложение будет доступно по адресу: http://localhost:3000

## Конфигурация

### Переменные окружения

Создайте `.env.local` файл:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Структура проекта

```
frontend/
├── src/
│   ├── app/              # Next.js App Router
│   │   ├── page.tsx      # Главная страница
│   │   ├── layout.tsx   # Layout приложения
│   │   └── globals.css  # Глобальные стили
│   ├── components/      # UI компоненты
│   │   └── ui/          # shadcn/ui компоненты
│   └── lib/             # Утилиты и API
│       └── api.ts       # API клиент
├── public/              # Статические файлы
├── package.json
├── next.config.ts
└── tailwind.config.ts
```

## Использование

### Выбор модели

1. Выберите модель из выпадающего списка
2. Нажмите "Обновить" для получения актуального списка
3. Нажмите "Загрузить модель" для загрузки выбранной модели

### Запуск инференса

1. Выберите изображение с помощью кнопки "Выберите файл"
2. Нажмите "Запустить инференс"
3. Просмотрите результаты:
   - Обнаруженные классы с уверенностью
   - Результат сегментации (overlay)
   - Маску сегментации

### Взаимодействие с классами

При наведении на класс в списке:
- Изображение обновляется, показывая только этот класс
- Отображается подсветка области класса

## API клиент

### Функции

```typescript
// Список доступных моделей
listModels(): Promise<string[]>

// Загрузка модели
loadModel(modelName: string): Promise<void>

// Получение статуса модели
getModelStatus(): Promise<ModelStatus>

// Запуск инференса
runInference(imageFile: File): Promise<InferenceResult>

// Проверка здоровья сервиса
checkHealth(): Promise<boolean>
```

### Типы данных

```typescript
interface ModelStatus {
  loaded: boolean;
  classes?: string[];
  device?: string;
}

interface DetectedClass {
  class: string;
  confidence: number;
}

interface InferenceResult {
  original_image: string;
  mask: string;
  overlay: string;
  detected_classes: DetectedClass[];
  class_overlays: Record<string, string>;
}
```

## UI Компоненты

Используется библиотека shadcn/ui:

- Card - Карточки для секций
- Button - Кнопки действий
- Input - Поле ввода файла
- Label - Метки полей

## Стилизация

Проект использует Tailwind CSS v4 с поддержкой:
- Тёмной темы
- Анимаций через tw-animate-css
- Кастомных шрифтов (Jost, Geist)

## Сборка

### Development
```bash
npm run dev
```

### Production
```bash
npm run build
npm start
```

### Linting
```bash
npm run lint
```