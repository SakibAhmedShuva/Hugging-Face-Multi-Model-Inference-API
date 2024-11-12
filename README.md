# Hugging-Face-Multi-Model-Inference-API

A Flask-based REST API for running inference with multiple Hugging Face NER (Named Entity Recognition) models simultaneously. This API allows you to combine results from multiple models with configurable preferences and confidence-based entity merging. You can import the JSON output from the API directly in Label Studio with all annotations.

## Features

- 🚀 Multi-model inference support
- 📊 Confidence-based entity merging
- 🎯 Model preference configuration
- 🔄 Automatic latest model detection
- 📁 Support for both file uploads and direct JSON input
- 🔍 Intelligent token span handling
- 📋 Detailed logging system
- ✅ Health check endpoint

## Prerequisites

- Python 3.8+
- pip package manager
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SakibAhmedShuva/Hugging-Face-Multi-Model-Inference-API.git
cd Hugging-Face-Multi-Model-Inference-API
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install flask transformers torch werkzeug numpy
```

## Project Structure

```
Hugging-Face-Multi-Model-Inference-API/
├── app.py              # Main Flask application
├── inference.py        # Inference logic and utilities
├── models/            # Directory for storing NER models
│   └── model1/       # Individual model directories
│   └── model2/
└── README.md
```

## Usage

1. Place your Hugging Face NER models in the `models/` directory.

2. Start the Flask application:
```bash
python app.py
```

The server will start on `http://localhost:5000`.

## API Endpoints

### 1. Prediction Endpoint

**Endpoint:** `POST /predict`

Accepts two types of inputs:

#### A. Multipart Form Data
```bash
curl -X POST http://localhost:5000/predict \
  -F "file=@input.json" \
  -F "model_paths=model1" \
  -F "model_paths=model2" \
  -F "model_preferences={\"PERSON\":0,\"ORG\":1}"
```

#### B. Direct JSON
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Objects": [
      {"Text": "John works at Microsoft."},
      {"Text": "Apple was founded by Steve Jobs."}
    ]
  }'
```

**Input Parameters:**
- `file` (optional): JSON file containing text objects
- `model_paths` (optional): List of model directories to use
- `model_preferences` (optional): JSON object mapping entity types to preferred model indices

**Response Format:**
```json
[
  {
    "id": 1,
    "annotations": [{
      "completed_by": 1,
      "result": [
        {
          "value": {
            "start": 0,
            "end": 4,
            "text": "John",
            "labels": ["PERSON"],
            "confidence": 0.95
          },
          "id": "label_uuid",
          "from_name": "label",
          "to_name": "text",
          "type": "labels",
          "origin": "manual"
        }
      ],
      "result_count": 1
    }],
    "data": {
      "text": "John works at Microsoft."
    }
  }
]
```

### 2. Health Check Endpoint

**Endpoint:** `GET /health`

```bash
curl http://localhost:5000/health
```

**Response:**
```json
{
  "status": "healthy"
}
```

## Model Preferences

You can specify which model should be preferred for specific entity types using the `model_preferences` parameter. For example:

```json
{
  "PERSON": 0,  // Prefer first model for PERSON entities
  "ORG": 1      // Prefer second model for ORG entities
}
```

## Error Handling

The API returns appropriate HTTP status codes:
- 200: Successful operation
- 400: Invalid input
- 404: Models not found
- 500: Server error

All errors include a JSON response with an "error" key containing the error message.

## Logging

The application includes comprehensive logging with the following features:
- Request details logging
- Model loading status
- Entity detection results
- Processing steps
- Error tracking

Logs are formatted with timestamps, logger name, and log level.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# Screenshot
![image](https://github.com/user-attachments/assets/2b8aed42-93c6-4c42-86d3-c7fcaa7e69f6)
