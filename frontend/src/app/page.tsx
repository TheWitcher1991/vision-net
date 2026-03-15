"use client";

import { useState, useEffect } from "react";
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
import {
  listModels,
  loadModel,
  getModelStatus,
  runInference,
  checkHealth,
  type InferenceResult,
} from "@/lib/api";

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
          <Card>
            <CardHeader>
              <CardTitle>Результаты</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {result.detected_classes.length > 0 ? (
                <div className="space-y-2">
                  <h3 className="font-medium">
                    Обнаруженные классы (наведите для подсветки):
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {result.detected_classes.map((cls, idx) => (
                      <span
                        key={idx}
                        onMouseEnter={() => setHoveredClass(cls.class)}
                        onMouseLeave={() => setHoveredClass(null)}
                        className="px-3 py-1 bg-secondary text-secondary-foreground rounded-full text-sm cursor-pointer hover:bg-primary hover:text-primary-foreground transition-colors"
                      >
                        {cls.class}: {(cls.confidence * 100).toFixed(1)}%
                      </span>
                    ))}
                  </div>
                </div>
              ) : (
                <p className="text-muted-foreground">Классы не обнаружены</p>
              )}

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <h3 className="font-medium mb-2">Результат</h3>
                  <img
                    src={`data:image/png;base64,${hoveredClass && result.class_overlays?.[hoveredClass] ? result.class_overlays[hoveredClass] : result.overlay}`}
                    alt="Результат"
                    className="w-full rounded-lg border"
                  />
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
        )}
      </div>
    </div>
  );
}
