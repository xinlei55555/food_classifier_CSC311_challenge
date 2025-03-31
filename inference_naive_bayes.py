import json
import numpy as np
from models.np_naive_bayes_utils import transform_vectorizer, make_inference

def load_model(model_dir='saved_model'):
    """Load trained model components from files"""
    # Load class priors
    with open(f'{model_dir}/class_priors.json', 'r') as f:
        class_priors = json.load(f)
    
    # Load class probabilities
    class_probs_loaded = np.load(f'{model_dir}/class_probs.npz')
    class_probs = {class_name: class_probs_loaded[class_name] for class_name in class_probs_loaded.files}
    
    # Load vocabulary
    with open(f'{model_dir}/vocab.json', 'r') as f:
        vocab = json.load(f)
    
    print(f"Model loaded from {model_dir}")
    return class_priors, class_probs, vocab

def predict(text, model_dir='saved_model', verbose=False):
    """Make prediction using saved model"""
    class_priors, class_probs, vocab = load_model(model_dir)
    print(text)
    return make_inference(class_priors, class_probs, vocab, text, verbose)

if __name__ == '__main__':
    # Example usage
    text = "I eat this with friends, on weekends, with little hot sauce"
    prediction, logits = predict(text, verbose=True)
    print(f"\nFinal Prediction: {prediction}")
    print(f"Logits: {logits}")
    
    text = 'I eat this with friends, on weekends, with little hot sauce, and at a party' 
    prediction, logits = predict(text, verbose=True)
    print(f"\nFinal Prediction: {prediction}")
    print(f"Logits: {logits}")