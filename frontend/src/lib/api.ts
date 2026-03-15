const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface ModelStatus {
  loaded: boolean;
  classes?: string[];
  device?: string;
}

export interface DetectedClass {
  class: string;
  confidence: number;
}

export interface InferenceResult {
  original_image: string;
  mask: string;
  overlay: string;
  detected_classes: DetectedClass[];
  class_overlays: Record<string, string>;
}

export async function listModels(): Promise<string[]> {
  const response = await fetch(`${API_BASE_URL}/models`);
  if (!response.ok) {
    throw new Error("Failed to list models");
  }
  const data = await response.json();
  return data.models;
}

export async function loadModel(modelName: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/model/load`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_name: modelName }),
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to load model");
  }
}

export async function getModelStatus(): Promise<ModelStatus> {
  const response = await fetch(`${API_BASE_URL}/model/status`);
  if (!response.ok) {
    throw new Error("Failed to get model status");
  }
  return response.json();
}

export async function runInference(imageFile: File): Promise<InferenceResult> {
  const formData = new FormData();
  formData.append("image", imageFile);

  const response = await fetch(`${API_BASE_URL}/infer`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to run inference");
  }
  return response.json();
}

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.ok;
  } catch {
    return false;
  }
}
