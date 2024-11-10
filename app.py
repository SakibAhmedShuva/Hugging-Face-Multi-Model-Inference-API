from flask import Flask
from inference import run_inference
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)

# Register the inference endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for running NER inference
    Accepts both multipart/form-data with JSON file and direct JSON POST requests
    """
    return run_inference()

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return {'status': 'healthy'}, 200

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(host='0.0.0.0', port=5000, debug=False)