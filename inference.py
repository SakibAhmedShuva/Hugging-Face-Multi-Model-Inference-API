from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import os
import datetime
import uuid
import json
import re
import numpy as np
import logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extend_token_spans(entity, text):
    """
    Extends entity spans to include adjacent numbers and special characters that should be part of the token.
    """
    start, end = entity['start'], entity['end']
    original_text = text[start:end]
    
    # Extend forward to include numbers and special chars
    while end < len(text) and (text[end].isdigit() or text[end] in '-_'):
        if text[start:end+1].lower() == original_text.lower() + text[end]:
            end += 1
        else:
            break
            
    # Extend backward to include numbers and special chars
    while start > 0 and (text[start-1].isdigit() or text[start-1] in '-_'):
        if text[start-1:end].lower() == text[start-1] + original_text.lower():
            start -= 1
        else:
            break
    
    return start, end

def load_model(model_path):
    """Load model and tokenizer from the given path."""
    logger.info(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    return pipeline('ner', grouped_entities=True, model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def merge_ner_results(results_list, text, model_preferences=None):
    logger.info("Starting merge_ner_results")
    logger.info(f"Model preferences: {model_preferences}")
    
    all_annotations = []
    for model_idx, results in enumerate(results_list):
        annotations = build_annotations(results, text)
        for annotation in annotations:
            annotation['model_idx'] = model_idx
        all_annotations.extend(annotations)
    
    # Build a list of all unique start and end positions
    positions = set()
    for annotation in all_annotations:
        if 'value' in annotation:
            positions.add(annotation['value']['start'])
            positions.add(annotation['value']['end'])
    positions = sorted(positions)

    # Create segments between positions
    segments = []
    for i in range(len(positions) - 1):
        segment_start = positions[i]
        segment_end = positions[i + 1]
        segment_text = text[segment_start:segment_end]
        segments.append({
            'start': segment_start,
            'end': segment_end,
            'text': segment_text,
            'annotations': []
        })

    # Assign annotations to segments
    for segment in segments:
        for annotation in all_annotations:
            anno_start = annotation['value']['start']
            anno_end = annotation['value']['end']
            if segment['start'] >= anno_start and segment['end'] <= anno_end:
                segment['annotations'].append(annotation)

    # For each segment, select the annotation with the highest confidence
    combined_annotations = []
    for segment in segments:
        if segment['annotations']:
            if model_preferences:
                preferred_annotations = [a for a in segment['annotations'] if str(a['model_idx']) == str(model_preferences.get(a['value']['labels'][0]))]
                if preferred_annotations:
                    best_annotation = max(preferred_annotations, key=lambda x: x['value'].get('confidence', 0))
                else:
                    best_annotation = max(segment['annotations'], key=lambda x: x['value'].get('confidence', 0))
            else:
                best_annotation = max(segment['annotations'], key=lambda x: x['value'].get('confidence', 0))
            
            new_annotation = best_annotation.copy()
            new_annotation['value'] = new_annotation['value'].copy()
            new_annotation['value']['start'] = segment['start']
            new_annotation['value']['end'] = segment['end']
            new_annotation['value']['text'] = segment['text']
            combined_annotations.append(new_annotation)

    logger.info(f"Merge completed. Total unique entities: {len(combined_annotations)}")
    return combined_annotations

def find_latest_model(models_dir='./models'):
    """Finds the latest model in the specified directory."""
    subdirs = [
        os.path.join(models_dir, d)
        for d in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, d))
    ]
    if not subdirs:
        raise FileNotFoundError(f"No models found in {models_dir}")
    latest_subdir = max(subdirs, key=os.path.getmtime)
    return latest_subdir

def parse_json_input(json_data):
    """Parses JSON input and extracts text from Objects array."""
    try:
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data
            
        texts = []
        if 'Objects' in data and isinstance(data['Objects'], list):
            for obj in data['Objects']:
                if 'Text' in obj:
                    texts.append(obj['Text'])
        return texts
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error parsing JSON: {str(e)}")

def merge_tokens_by_original_text(annotations, original_text):
    """Merge divided tokens back into single entities based on the original text tokens."""
    # Split original text into words/tokens
    original_tokens = re.findall(r'\b\w+\b', original_text)
    final_annotations = []
    i = 0

    for token in original_tokens:
        token_start = original_text.find(token, i)
        token_end = token_start + len(token)
        
        # Find all annotations that overlap with the token
        overlapping_annotations = [
            annotation for annotation in annotations 
            if annotation['value']['start'] >= token_start and annotation['value']['end'] <= token_end
        ]
        
        # If there are overlapping annotations, take the one with the highest confidence
        if overlapping_annotations:
            highest_confidence_annotation = max(overlapping_annotations, key=lambda a: a['value']['confidence'])
            highest_confidence_annotation['value']['start'] = token_start
            highest_confidence_annotation['value']['end'] = token_end
            highest_confidence_annotation['value']['text'] = token
            
            final_annotations.append(highest_confidence_annotation)
        
        # Move index to end of the current token
        i = token_end

    return final_annotations

def build_annotations(ner_results, text):
    """Builds annotations in the required format from NER results, handling combined tokens properly."""
    
    # Sort results by start position
    sorted_results = sorted(ner_results, key=lambda x: x.get('start', 0))
    
    # Initialize combined results
    combined_results = []
    current_entity = None
    
    def should_combine(curr, next_entity):
        """Helper function to determine if entities should be combined"""
        if curr is None or next_entity is None:
            return False
            
        # Check if same entity type
        if curr.get('entity_group') != next_entity.get('entity_group'):
            return False
            
        # Get the text between entities
        text_between = text[curr['end']:next_entity['start']]
        
        # Check if entities are adjacent or separated only by whitespace/special chars
        return bool(re.match(r'^[\s\d_-]*$', text_between))
    
    # Combine consecutive entities with same label
    for entity in sorted_results:
        if not all(key in entity for key in ['start', 'end', 'score']):
            continue
        
        # Extend spans for the current entity
        entity['start'], entity['end'] = extend_token_spans(entity, text)
        
        if current_entity is None:
            current_entity = entity.copy()
            current_entity['score'] = float(current_entity['score'])
            continue
            
        # Check if entities should be combined
        if should_combine(current_entity, entity):
            # Extend current entity
            current_entity['end'] = entity['end']
            # Update text to include the combined span
            current_entity['word'] = text[current_entity['start']:current_entity['end']]
            # Update score to highest of combined entities
            current_entity['score'] = max(float(current_entity['score']), float(entity['score']))
        else:
            # Add current entity to results and start new one
            combined_results.append(current_entity)
            current_entity = entity.copy()
            current_entity['score'] = float(current_entity['score'])
    
    # Add the last entity if exists
    if current_entity is not None:
        combined_results.append(current_entity)
    
    # Post-process to extend number spans
    def extend_number_span(start, end, text):
        """Helper function to extend span to include full numbers"""
        # Extend start backward if in middle of number
        while start > 0 and text[start-1].isdigit():
            start -= 1
            
        # Extend end forward if in middle of number
        while end < len(text) and text[end].isdigit():
            end += 1
            
        return start, end
    
    # Apply number span extension to results
    for entity in combined_results:
        if any(c.isdigit() for c in text[entity['start']:entity['end']]):
            entity['start'], entity['end'] = extend_number_span(
                entity['start'], 
                entity['end'],
                text
            )
    
    # Build annotations from combined results
    annotations = []
    for entity in combined_results:
        score = float(entity.get('score', 0))
        confidence = round(score * 100, 2)
        
        # Get the full text span
        entity_text = text[entity['start']:entity['end']]
        
        annotation = {
            "value": {
                "start": entity['start'],
                "end": entity['end'],
                "text": entity_text,
                "labels": [entity.get('entity_group') or entity.get('entity', 'UNKNOWN')],
                "confidence": confidence
            },
            "id": f"label_{uuid.uuid4()}",
            "from_name": "label",
            "to_name": "text",
            "type": "labels",
            "origin": "manual"
        }
        annotations.append(annotation)

    return annotations

def run_inference():
    logger.info("Starting inference")
    try:
        # Log request details
        logger.info("Request details:")
        if 'file' in request.files:
            file = request.files['file']
            logger.info(f"Input file name: {file.filename}")
            file_contents = file.read().decode('utf-8')
            texts = parse_json_input(file_contents)
        elif request.is_json:
            logger.info("Direct JSON input received")
            texts = parse_json_input(request.get_json())
        else:
            return jsonify({"error": "No valid input provided"}), 400

        # Log the texts to be processed
        logger.info(f"Number of texts to process: {len(texts)}")
        for idx, text in enumerate(texts):
            logger.info(f"Text {idx + 1}: {text[:100]}...")  # Log first 100 chars of each text

        # Get model paths and preferences from the request
        model_paths = request.form.getlist('model_paths')
        logger.info(f"Model paths received: {model_paths}")
        
        try:
            model_preferences = json.loads(request.form.get('model_preferences', '{}'))
            logger.info(f"Model preferences received: {model_preferences}")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing model preferences: {str(e)}")
            model_preferences = {}

        # If no model paths provided, use the latest model
        if not model_paths:
            try:
                model_paths = [find_latest_model()]
                logger.info(f"Using latest model: {model_paths[0]}")
            except FileNotFoundError as e:
                return jsonify({"error": str(e)}), 404

        # Load all specified models
        nlp_models = []
        for idx, path in enumerate(model_paths):
            # Check if path already starts with ./models
            if path.startswith('./models'):
                full_path = path
            else:
                full_path = os.path.join('./models', path)
                
            # Normalize path to handle any double slashes or incorrect separators
            full_path = os.path.normpath(full_path)
            
            if not os.path.exists(full_path):
                logger.warning(f"Model path {full_path} does not exist.")
                continue
            
            logger.info(f"Loading model {idx + 1}: {full_path}")
            model = load_model(full_path)
            nlp_models.append(model)
            logger.info(f"Successfully loaded model {idx + 1}")

        if not nlp_models:
            return jsonify({"error": "No valid models found"}), 404

        # Run inference and build results
        results = []
        for idx, text in enumerate(texts):
            logger.info(f"\nProcessing text {idx + 1}")
            
            # Run inference with all models
            all_ner_results = []
            for model_idx, model in enumerate(nlp_models):
                logger.info(f"Running model {model_idx + 1} on text {idx + 1}")
                ner_result = model(text)
                logger.info(f"Model {model_idx + 1} found {len(ner_result)} entities")
                all_ner_results.append(ner_result)
            
            # Merge results from all models using preferences
            logger.info("Merging results from all models")
            merged_annotations = merge_ner_results(all_ner_results, text, model_preferences)
            logger.info(f"After merging: {len(merged_annotations)} unique entities")
            
            annotation_entry = {
                "completed_by": 1,
                "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                "ground_truth": False,
                "id": int(uuid.uuid4().int & (1<<16)-1),
                "lead_time": 0.0,
                "prediction": {},
                "project": 3,
                "result": merged_annotations,
                "result_count": len(merged_annotations),
                "task": int(uuid.uuid4().int & (1<<16)-1),
                "unique_id": str(uuid.uuid4()),
                "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
                "was_cancelled": False
            }
            
            output_entry = {
                "id": idx + 1,
                "annotations": [annotation_entry],
                "data": {"text": text},
                "meta": {},
                "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
                "inner_id": idx + 1,
                "total_annotations": 1,
                "cancelled_annotations": 0,
                "total_predictions": 0,
                "comment_count": 0,
                "unresolved_comment_count": 0,
                "project": 3
            }
            results.append(output_entry)

        logger.info("Inference completed successfully")
        return jsonify(results)

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        return jsonify({"error": str(e)}), 500