import json
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import clean.drink_cleaning
from models.np_naive_bayes_utils import transform_vectorizer, make_inference

common_drinks = clean.drink_cleaning.parse_common_drinks(os.path.join("clean", "common_drinks.simple"))

def load_model(model_dir='saved_model', verbose=False):
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
    if verbose:
        print(f"Model loaded from {model_dir}")
    return class_priors, class_probs, vocab

def predict(text, model_dir='saved_model', verbose=False):
    """Make prediction using saved model"""
    class_priors, class_probs, vocab = load_model(model_dir, verbose)
    return make_inference(class_priors, class_probs, vocab, text, verbose)

def predict_smart(data: list, model_dir='saved_model', verbose=False):
    """Make prediction using saved model"""
    class_priors, class_probs, vocab = load_model(model_dir, verbose)
    data[6] = clean.drink_cleaning.process_drink(data[6], common_drinks)
    unwanted_indexes = [0, 1, 2, 4]
    np.delete(data, unwanted_indexes, axis=0)
    return make_inference(class_priors, class_probs, vocab, " ".join(data), verbose)

if __name__ == '__main__':
    # Example usage
    # text = "I eat this with friends, on weekends, with little hot sauce"
    # prediction, logits = predict(text, verbose=True)
    # print(f"\nFinal Prediction: {prediction}")
    # print(f"Logits: {logits}")
    
    # text = 'I eat this with friends, on weekends, with little hot sauce, and at a party' 
    # prediction, logits = predict(text, verbose=True)
    # print(f"\nFinal Prediction: {prediction}")
    # print(f"Logits: {logits}")

    # text = 'I eat this with friends, on weekends, with no hot sauce, and at dinner, and it makes me think of tokyo drift.'
    # prediction, logits = predict(text, verbose=True)
    # print(f"\nFinal Prediction: {prediction}")
    # print(f"Logits: {logits}")

    # # data = ['I eat this with friends, on weekends, with little hot sauce, and at a party', 'sdlkfj'] 
    # data = ['716549','3','6',"Week day lunch,At a party,Late night snack",'5','Cloudy with a Chance of Meatballs','Coke','Friends','A little (mild)']
    # print(" ".join(data), "-> aa",)
    # prediction, logits = predict_smart(data, verbose=True)
    # print(f"\nFinal Prediction: {prediction}")
    # print(f"Logits: {logits}")


    # text = 'This item would cost around 5 dollars, have 5 ingredients, and be eaten at a party with friends.'
    # text = "'999993', '3', '7', 'Weekend lunch,Weekend dinner,At a party,Late night snack', '7$', 'Spaceballs and other parody movies', 'carbonated sweet drinks, such as cocacola, pepsi, etc', 'Friends', 'None',"
    # text = "'5', '4', 'Week day lunch,Week day dinner,Weekend lunch,Weekend dinner,At a party', '15', 'Cars 2', 'Water', 'Friends,Strangers', 'None'"
    text = "'4', '9', 'Week day lunch,Week day dinner,Late night snack', '11.99', 'borat', 'Soave', 'Friends,Strangers', 'A lot (hot)'"
    prediction, logits = predict(text, verbose=True)
    print(f"\nFinal Prediction: {prediction}")
    print(f"Logits: {logits}")
