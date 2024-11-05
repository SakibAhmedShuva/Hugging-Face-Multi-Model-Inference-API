```markdown
# Hugging Face Multi-Model Inference API

## Overview

The Multi-Model Inference API is a Flask-based service that provides Named Entity Recognition (NER) capabilities using multiple pre-trained models. It allows for flexible model selection, result merging, and preference-based entity selection.

## Features

- Support for multiple NER models
- Automatic latest model detection
- Model preference configuration
- JSON input parsing
- Detailed logging
- Error handling and reporting

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/SakibAhmedShuva/Hugging-Face-Multi-Model-Inference-API.git
   cd Hugging-Face-Multi-Model-Inference-API
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Place your pre-trained models in the `./models` directory.

## Usage

1. Start the Flask server:
   ```
   python app.py
   ```

2. The API will be available at `http://localhost:5000/`.

## API Endpoints

### POST /inference

Performs NER inference on the provided text using specified models.

#### Request

- Content-Type: `multipart/form-data`
- Body:
  - `file`: JSON file containing text to analyze (optional)
  - `model_paths`: List of model paths to use (optional)
  - `model_preferences`: JSON string of model preferences (optional)

If no file is provided, the API expects a JSON payload in the request body.

#### Response

JSON array of annotation results for each input text.

## Postman Documentation

### Request

1. Open Postman and create a new request.
2. Set the request type to POST.
3. Enter the URL: `http://localhost:5000/inference`
4. In the "Body" tab, select "form-data".
5. Add the following key-value pairs:
   - Key: `file`, Value: Select your JSON file
   - Key: `model_paths`, Value: `./models/model1,./models/model2` (comma-separated list of model paths)
   - Key: `model_preferences`, Value: `{"PERSON": "0", "ORG": "1"}` (JSON string of label preferences)

### Response

The API will return a JSON array of annotation results. Each result will contain:

- Annotation details (id, created_at, updated_at, etc.)
- Recognized entities with their positions, labels, and confidence scores
- Original input text

## Error Handling

The API includes comprehensive error handling and logging. In case of errors, it will return a JSON response with an "error" key containing the error message.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
