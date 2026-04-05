"use client";

import { useState, useEffect, useMemo } from "react";
import { cn } from "@/lib/utils";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import {
  listModels,
  loadModel,
  getModelStatus,
  runInference,
  checkHealth,
  type InferenceResult,
  type BoundingBox,
} from "@/lib/api";

interface ClassStats {
  class: string;
  count: number;
  avgConfidence: number;
  minConfidence: number;
  maxConfidence: number;
}

export default function InferencePage() {
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [isHealthy, setIsHealthy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hoveredClass, setHoveredClass] = useState<string | null>(null);
  const [focusEnabled, setFocusEnabled] = useState(false);
  const [selectedClasses, setSelectedClasses] = useState<Set<string>>(
    new Set(),
  );

  const [imageDimensions, setImageDimensions] = useState<{
    width: number;
    height: number;
  } | null>(null);

  const classStats = useMemo<ClassStats[]>(() => {
    if (!result?.detected_classes) return [];

    const classMap = new Map<string, { confidences: number[] }>();

    for (const cls of result.detected_classes) {
      if (!classMap.has(cls.class)) {
        classMap.set(cls.class, { confidences: [] });
      }
      classMap.get(cls.class)!.confidences.push(cls.confidence);
    }

    return Array.from(classMap.entries()).map(([className, data]) => ({
      class: className,
      count: data.confidences.length,
      avgConfidence:
        data.confidences.reduce((a, b) => a + b, 0) / data.confidences.length,
      minConfidence: Math.min(...data.confidences),
      maxConfidence: Math.max(...data.confidences),
    }));
  }, [result]);

  const allClasses = useMemo(() => {
    if (!result?.detected_classes) return [];
    return [...new Set(result.detected_classes.map((c) => c.class))];
  }, [result]);

  const filteredClasses = useMemo(() => {
    if (selectedClasses.size === 0) return result?.detected_classes || [];
    return (
      result?.detected_classes.filter((c) => selectedClasses.has(c.class)) || []
    );
  }, [result, selectedClasses]);

  const toggleClass = (cls: string) => {
    const newSet = new Set(selectedClasses);
    if (newSet.has(cls)) {
      newSet.delete(cls);
    } else {
      newSet.add(cls);
    }
    setSelectedClasses(newSet);
  };

  const selectAllClasses = () => {
    setSelectedClasses(new Set(allClasses));
  };

  const clearClassSelection = () => {
    setSelectedClasses(new Set());
  };

  const imageTransform = useMemo(() => {
    if (!focusEnabled || !hoveredClass || !result || !imageDimensions)
      return null;

    const cls = result.detected_classes.find((c) => c.class === hoveredClass);
    if (!cls?.bbox) return null;

    const bbox = cls.bbox;
    const imgWidth = imageDimensions.width;
    const imgHeight = imageDimensions.height;

    const centerX = (bbox.x_min + bbox.x_max) / 2;
    const centerY = (bbox.y_min + bbox.y_max) / 2;
    const bboxWidth = bbox.x_max - bbox.x_min;
    const bboxHeight = bbox.y_max - bbox.y_min;

    const scale = Math.min(imgWidth / bboxWidth, imgHeight / bboxHeight, 2.5);

    const translateX = 50 - (centerX / imgWidth) * 100 * scale;
    const translateY = 50 - (centerY / imgHeight) * 100 * scale;

    return {
      transform: `translate(${translateX}%, ${translateY}%) scale(${scale})`,
      transformOrigin: "center center",
    };
  }, [focusEnabled, hoveredClass, result, imageDimensions]);

  const handleListModels = async () => {
    try {
      const modelList = await listModels();
      setModels(modelList);
    } catch (e) {
      console.error("Ошибка при получении списка моделей:", e);
    }
  };

  useEffect(() => {
    checkHealth().then(setIsHealthy);
    getModelStatus()
      .then((status) => setIsModelLoaded(status.loaded))
      .catch(() => {});
    handleListModels();
  }, []);

  const handleLoadModel = async () => {
    if (!selectedModel) {
      setError("Пожалуйста, выберите модель");
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      await loadModel(selectedModel);
      setIsModelLoaded(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Ошибка при загрузке модели");
    } finally {
      setIsLoading(false);
    }
  };

  const handleInference = async () => {
    if (!selectedFile) {
      setError("Пожалуйста, выберите изображение");
      return;
    }
    setIsLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await runInference(selectedFile);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Ошибка при запуске инференса");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background p-8">
      <div className="max-w-4xl mx-auto space-y-8">
        <div className="text-center">
          <h1 className="text-3xl font-bold">VisionNet Инференс</h1>
          <p className="text-muted-foreground mt-2">
            Клиент API семантической сегментации
          </p>
        </div>

        {!isHealthy && (
          <Card className="border-destructive">
            <CardContent className="pt-6">
              <p className="text-destructive">
                Сервер недоступен. Убедитесь, что бэкенд запущен на порту 8000.
              </p>
            </CardContent>
          </Card>
        )}

        {error && (
          <Card className="border-destructive">
            <CardContent className="pt-6">
              <p className="text-destructive">{error}</p>
            </CardContent>
          </Card>
        )}

        <Card>
          <CardHeader>
            <CardTitle>Выбор модели</CardTitle>
            <CardDescription>
              Выберите модель из папки backend/app/models
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center gap-4">
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
              >
                <option value="">Выберите модель...</option>
                {models.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
              <Button onClick={handleListModels} variant="outline">
                Обновить
              </Button>
            </div>
            <div className="flex items-center gap-4">
              <Button
                onClick={handleLoadModel}
                disabled={isLoading || !selectedModel}
              >
                {isLoading ? "Загрузка..." : "Загрузить модель"}
              </Button>
              {isModelLoaded && (
                <span className="text-green-600 text-sm">Модель загружена</span>
              )}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Запуск инференса</CardTitle>
            <CardDescription>
              Загрузите изображение для семантической сегментации
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="image">Файл изображения</Label>
              <Input
                id="image"
                type="file"
                accept="image/*"
                onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
              />
            </div>
            <Button
              onClick={handleInference}
              disabled={isLoading || !isModelLoaded || !selectedFile}
            >
              {isLoading ? "Обработка..." : "Запустить инференс"}
            </Button>
          </CardContent>
        </Card>

        {result && (
          <>
            <Card>
              <CardHeader>
                <CardTitle>Статистика по классам</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left py-2 px-3 font-medium">
                          Класс
                        </th>
                        <th className="text-right py-2 px-3 font-medium">
                          Кол-во
                        </th>
                        <th className="text-right py-2 px-3 font-medium">
                          Среднее
                        </th>
                        <th className="text-right py-2 px-3 font-medium">
                          Мин
                        </th>
                        <th className="text-right py-2 px-3 font-medium">
                          Макс
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {classStats.map((stat) => (
                        <tr
                          key={stat.class}
                          className="border-b hover:bg-muted/50"
                        >
                          <td className="py-2 px-3">
                            <Badge variant="secondary">{stat.class}</Badge>
                          </td>
                          <td className="text-right py-2 px-3 font-mono">
                            {stat.count}
                          </td>
                          <td className="text-right py-2 px-3 font-mono">
                            {(stat.avgConfidence * 100).toFixed(1)}%
                          </td>
                          <td className="text-right py-2 px-3 font-mono">
                            {(stat.minConfidence * 100).toFixed(1)}%
                          </td>
                          <td className="text-right py-2 px-3 font-mono">
                            {(stat.maxConfidence * 100).toFixed(1)}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="mt-4 flex items-center gap-4 text-sm text-muted-foreground">
                  <span>
                    Всего объектов:{" "}
                    <span className="font-mono font-medium text-foreground">
                      {result.detected_classes.length}
                    </span>
                  </span>
                  <span>
                    Классов:{" "}
                    <span className="font-mono font-medium text-foreground">
                      {classStats.length}
                    </span>
                  </span>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Результаты</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {result.detected_classes.length > 0 ? (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between flex-wrap gap-2">
                      <h3 className="font-medium">
                        Классы ({filteredClasses.length} /{" "}
                        {result.detected_classes.length}):
                      </h3>
                      <div className="flex items-center gap-4">
                        <div className="flex items-center gap-2">
                          <span className="text-sm text-muted-foreground">
                            Фильтр
                          </span>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={selectAllClasses}
                          >
                            Все
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={clearClassSelection}
                          >
                            Сбросить
                          </Button>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-sm text-muted-foreground">
                            Фокус
                          </span>
                          <Switch
                            checked={focusEnabled}
                            onCheckedChange={setFocusEnabled}
                          />
                        </div>
                      </div>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {allClasses.map((cls) => {
                        const isSelected =
                          selectedClasses.size === 0 ||
                          selectedClasses.has(cls);
                        const stat = classStats.find((s) => s.class === cls);
                        const isHovered = hoveredClass === cls;
                        return (
                          <Badge
                            key={cls}
                            variant={isSelected ? "default" : "outline"}
                            className={cn(
                              "cursor-pointer hover:opacity-80 transition-all",
                              isHovered && "ring-primary",
                            )}
                            onClick={() => toggleClass(cls)}
                            onMouseEnter={() => setHoveredClass(cls)}
                            onMouseLeave={() => setHoveredClass(null)}
                          >
                            {cls}:{" "}
                            {((stat?.avgConfidence ?? 0) * 100).toFixed(0)}% (
                            {stat?.count || 0})
                          </Badge>
                        );
                      })}
                    </div>
                  </div>
                ) : (
                  <p className="text-muted-foreground">Классы не обнаружены</p>
                )}

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <h3 className="font-medium mb-2">
                      Результат (фокусировка: {hoveredClass || "нет"})
                    </h3>
                    <div className="overflow-hidden rounded-lg border">
                      <img
                        src={`data:image/png;base64,${hoveredClass && result.class_overlays?.[hoveredClass] ? result.class_overlays[hoveredClass] : result.overlay}`}
                        alt="Результат"
                        className="w-full rounded-lg border transition-transform duration-300"
                        style={imageTransform || undefined}
                        onLoad={(e) => {
                          const img = e.target as HTMLImageElement;
                          setImageDimensions({
                            width: img.naturalWidth,
                            height: img.naturalHeight,
                          });
                        }}
                        onMouseMove={(e) => {
                          if (!imageDimensions) return;
                          const rect = e.currentTarget.getBoundingClientRect();
                          const x =
                            ((e.clientX - rect.left) / rect.width) *
                            imageDimensions.width;
                          const y =
                            ((e.clientY - rect.top) / rect.height) *
                            imageDimensions.height;

                          for (const cls of filteredClasses) {
                            const bbox = cls.bbox;
                            if (
                              x >= bbox.x_min &&
                              x <= bbox.x_max &&
                              y >= bbox.y_min &&
                              y <= bbox.y_max
                            ) {
                              setHoveredClass(cls.class);
                              return;
                            }
                          }
                          setHoveredClass(null);
                        }}
                        onMouseLeave={() => setHoveredClass(null)}
                      />
                    </div>
                  </div>
                  <div>
                    <h3 className="font-medium mb-2">Маска сегментации</h3>
                    <img
                      src={`data:image/png;base64,${result.mask}`}
                      alt="Маска"
                      className="w-full rounded-lg border"
                    />
                  </div>
                </div>
              </CardContent>
            </Card>
          </>
        )}
      </div>
    </div>
  );
}
