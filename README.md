# CNN-Deployment

FastAPI application for blood-pressure monitor brand classification using CNN models.

## Features

- **Multi-model support**: 5 pre-trained Keras models (EfficientNetV2B0, MobileNetV3Large, DenseNet121, ConvNeXtTiny, NASNetMobile)
- **Image upload**: Upload images via `/upload` endpoint and get a public URL
- **Prediction**: Classify images by brand with confidence scores
- **5 Classes**: Allwell, Lifebox, Omron, Sinocare, Yuwell

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/get-models` | List all available models with IDs |
| `POST` | `/upload` | Upload image file, returns public URL |
| `POST` | `/predict` | Predict image class using model |

### Example Usage

#### 1. Get available models
```bash
curl http://localhost:8000/get-models
```

Response:
```json
{
  "models": [
    {"id": "efficientnetv2b0", "name": "EfficientNetV2B0"},
    {"id": "mobilenetv3large", "name": "MobileNetV3Large"},
    ...
  ]
}
```

#### 2. Upload an image
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/image.jpg"
```

Response:
```json
{
  "url": "http://localhost:8000/uploads/abc123.jpg",
  "filename": "image.jpg",
  "size": 45231
}
```

#### 3. Predict with uploaded image
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "http://localhost:8000/uploads/abc123.jpg",
    "model_id": "efficientnetv2b0"
  }'
```

Response:
```json
{
  "model_id": "efficientnetv2b0",
  "model_name": "EfficientNetV2B0",
  "image_url": "http://localhost:8000/uploads/abc123.jpg",
  "predicted_class": "Omron",
  "confidence": 0.9543,
  "probabilities": [
    {"class_name": "Allwell", "confidence": 0.0123},
    {"class_name": "Lifebox", "confidence": 0.0089},
    {"class_name": "Omron", "confidence": 0.9543},
    {"class_name": "Sinocare", "confidence": 0.0156},
    {"class_name": "Yuwell", "confidence": 0.0089}
  ]
}
```

## Local run

1. Create a virtualenv and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Configure environment variables (optional):

   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env to customize settings:
   # - CORS_ORIGINS: Comma-separated list of allowed origins
   # - MAX_UPLOAD_SIZE: Maximum file size in bytes
   # - LOG_LEVEL: Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
   ```

3. Start the API:

   ```bash
   uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000
   ```

4. Visit API docs:
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

## Configuration

The application uses environment variables for configuration. Create a `.env` file or set environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CORS_ORIGINS` | `http://localhost:3000,...` | Comma-separated allowed origins. Use `*` for all (dev only) |
| `CORS_ALLOW_CREDENTIALS` | `true` | Allow cookies and auth headers |
| `CORS_ALLOW_METHODS` | `GET,POST,PUT,DELETE,OPTIONS` | Allowed HTTP methods |
| `CORS_ALLOW_HEADERS` | `*` | Allowed request headers |
| `MAX_UPLOAD_SIZE` | `10485760` (10 MB) | Maximum file upload size in bytes |
| `ALLOWED_EXTENSIONS` | `.jpg,.jpeg,.png,.webp` | Comma-separated file extensions |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |

**Example `.env` file:**
```env
CORS_ORIGINS=http://localhost:3000,https://myapp.com
CORS_ALLOW_CREDENTIALS=true
MAX_UPLOAD_SIZE=5242880
LOG_LEVEL=DEBUG
```


## Docker

- Build and run with Compose:
  ```bash
  docker compose up --build
  ```

## Tests

- Run unit tests:
  ```bash
  pytest
  ```

## Project Structure

```
src/
├── app/
│   ├── config.py           # Configuration & model registry
│   ├── main.py             # FastAPI app with lifespan
│   ├── router/
│   │   ├── health.py       # Health check
│   │   ├── models.py       # GET /get-models
│   │   ├── upload.py       # POST /upload
│   │   └── predict.py      # POST /predict
│   ├── schemas/            # Pydantic models
│   └── services/
│       └── model_service.py # Model loading & inference
├── models/                  # .keras model weights
└── uploads/                 # Uploaded images (gitignored)
```
