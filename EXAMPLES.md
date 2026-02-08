# API Usage Examples

## Frontend Integration Example (TypeScript/React)

```typescript
const API_URL = "http://localhost:8000";

interface PredictionResponse {
  model_id: string;
  model_name: string;
  image_url: string;
  predicted_class: string;
  confidence: number;
  probabilities: Array<{
    class_name: string;
    confidence: number;
  }>;
}

async function predictImage(
  selectedFile: File,
  selectedModel: string
): Promise<PredictionResponse> {
  try {
    // Step 1: Upload the image file
    const formData = new FormData();
    formData.append("file", selectedFile);

    const uploadResponse = await fetch(`${API_URL}/upload`, {
      method: "POST",
      body: formData,
    });

    if (!uploadResponse.ok) {
      throw new Error("Failed to upload image");
    }

    const uploadData = await uploadResponse.json();
    const imageUrl = uploadData.url;

    // Step 2: Make prediction request
    const predictionResponse = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        image_url: imageUrl,
        model_id: selectedModel,
      }),
    });

    if (!predictionResponse.ok) {
      throw new Error("Prediction failed");
    }

    const predictionData: PredictionResponse = await predictionResponse.json();
    return predictionData;
  } catch (error) {
    console.error("Error:", error);
    throw error;
  }
}

// Usage in React component
function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async () => {
    if (!selectedFile || !selectedModel) return;

    setIsLoading(true);
    try {
      const result = await predictImage(selectedFile, selectedModel);
      setPrediction(result);
    } catch (error) {
      console.error("Prediction failed:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div>
      <input
        type="file"
        accept="image/*"
        onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
      />
      <select
        value={selectedModel}
        onChange={(e) => setSelectedModel(e.target.value)}
      >
        <option value="">Select a model</option>
        <option value="efficientnetv2b0">EfficientNetV2B0</option>
        <option value="mobilenetv3large">MobileNetV3Large</option>
        <option value="densenet121">DenseNet121</option>
        <option value="convnexttiny">ConvNeXtTiny</option>
        <option value="nasnetmobile">NASNetMobile</option>
      </select>
      <button onClick={handleSubmit} disabled={isLoading}>
        {isLoading ? "Processing..." : "Predict"}
      </button>

      {prediction && (
        <div>
          <h3>Result: {prediction.predicted_class}</h3>
          <p>Confidence: {(prediction.confidence * 100).toFixed(2)}%</p>
          <p>Model: {prediction.model_name}</p>
          <img src={prediction.image_url} alt="Uploaded" width="300" />
        </div>
      )}
    </div>
  );
}
```

## Python Client Example

```python
import requests

API_URL = "http://localhost:8000"

def predict_image(image_path: str, model_id: str) -> dict:
    """Upload and predict an image."""
    
    # Step 1: Upload image
    with open(image_path, "rb") as f:
        files = {"file": f}
        upload_response = requests.post(f"{API_URL}/upload", files=files)
        upload_response.raise_for_status()
    
    upload_data = upload_response.json()
    image_url = upload_data["url"]
    
    # Step 2: Predict
    predict_response = requests.post(
        f"{API_URL}/predict",
        json={"image_url": image_url, "model_id": model_id}
    )
    predict_response.raise_for_status()
    
    return predict_response.json()


# Usage
if __name__ == "__main__":
    result = predict_image("./test_image.jpg", "efficientnetv2b0")
    print(f"Predicted: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    
    print("\nAll probabilities:")
    for prob in result["probabilities"]:
        print(f"  {prob['class_name']}: {prob['confidence']:.2%}")
```

## cURL Examples

### 1. Get available models
```bash
curl http://localhost:8000/get-models | jq
```

### 2. Upload an image
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@./my_image.jpg" | jq
```

### 3. Predict (after upload)
```bash
IMAGE_URL="http://localhost:8000/uploads/abc123.jpg"

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"image_url\": \"$IMAGE_URL\", \"model_id\": \"efficientnetv2b0\"}" | jq
```

### 4. Full workflow (upload + predict)
```bash
# Upload
UPLOAD_RESPONSE=$(curl -s -X POST http://localhost:8000/upload \
  -F "file=@./test_image.jpg")
IMAGE_URL=$(echo $UPLOAD_RESPONSE | jq -r '.url')

echo "Uploaded to: $IMAGE_URL"

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"image_url\": \"$IMAGE_URL\", \"model_id\": \"efficientnetv2b0\"}" | jq
```

## JavaScript/Fetch Example

```javascript
async function uploadAndPredict(file, modelId) {
  // Upload
  const formData = new FormData();
  formData.append('file', file);
  
  const uploadRes = await fetch('http://localhost:8000/upload', {
    method: 'POST',
    body: formData
  });
  const { url: imageUrl } = await uploadRes.json();
  
  // Predict
  const predictRes = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_url: imageUrl, model_id: modelId })
  });
  
  return await predictRes.json();
}

// Usage
const fileInput = document.querySelector('input[type="file"]');
const result = await uploadAndPredict(fileInput.files[0], 'efficientnetv2b0');
console.log('Prediction:', result.predicted_class);
```
