import json
import os


class ConfigManager:
    def __init__(self, config_path=None):
        self.config_path = config_path or "app/configs/model_config.json"
        self.config = self._default_config()

    def _default_config(self):
        return {
            "model_name": "VisionNetAdapter",
            "in_channels": 3,
            "num_classes": 9,
            "image_size": 512,
            "classes": [
                "stem",
                "flower",
                "fruit",
                "powdery_mildew_severity_1",
                "powdery_mildew_severity_2",
                "powdery_mildew_severity_3",
                "powdery_mildew_severity_4",
                "powdery_mildew_severity_5",
                "powdery_mildew_severity_6",
            ],
        }

    def load(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                self.config = json.load(f)
        return self.config

    def save(self, config=None):
        if config:
            self.config = config

        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

        with open(self.config_path, "w") as f:
            json.dump(self.config, f, indent=4)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value

    @staticmethod
    def create_model_config(model_name, in_channels, num_classes, image_size, classes):
        return {
            "model_name": model_name,
            "in_channels": in_channels,
            "num_classes": num_classes,
            "image_size": image_size,
            "classes": classes,
        }

    @staticmethod
    def save_model_config(config, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(config, f, indent=4)

    @staticmethod
    def load_model_config(path):
        with open(path, "r") as f:
            return json.load(f)
