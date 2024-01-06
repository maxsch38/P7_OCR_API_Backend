######################################################################################################################################################################
### Fichier API avec FastAPI 
######################################################################################################################################################################

######################################################################################################################################################################
### Importation des librairies : 
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import pandas as pd
import pickle
import warnings
import shap
import gzip

warnings.filterwarnings('ignore', category=FutureWarning)

###########################################################################################################################
# Fonction de gestion des données pickle et gzip: 

def chargement_pickle(name, chemin): 
    path = chemin + '/' + name + '.pickle'

    with open(path, 'rb') as f:
        fichier = pickle.load(f)

    return fichier


def enregistrement_pickle(name, chemin, fichier):    
    path = chemin + '/' + name + '.pickle'

    with open(path, 'wb') as f:
        pickle.dump(fichier, f)
        
def chargement_pickle_gzip(name, chemin): 
    path = chemin + '/' + name + '.pickle.gz'

    with gzip.open(path, 'rb') as f:
        fichier = pickle.load(f)

    return fichier

def enregistrement_pickle_gzip(name, chemin, fichier):    
    path = chemin + '/' + name + '.pickle.gz'

    with gzip.open(path, 'wb') as f:
        pickle.dump(fichier, f)
        
        
######################################################################################################################################################################
### Chargement des données : 

# Chemin du dossier sauvegarde : 
dossier_sauvegarde = 'Data'

# Pipeline : 
pipeline = chargement_pickle(
    name='Pipeline_DecisionTreeClassifier',
    chemin=dossier_sauvegarde,
)

# Threshold spécifique : 
threshold = chargement_pickle(
    name='Threshold_DecisionTreeClassifier',
    chemin=dossier_sauvegarde, 
)

# Lancement de l'API : 
app = FastAPI()

######################################################################################################################################################################
### Pydantic :
    
class UserInputPredict(BaseModel):
    X: Dict[str, float]
    
######################################################################################################################################################################
### Fonction get_threshold :

@app.get("/get_threshold")
def get_threshold():
    return {'Threshold': threshold}
 
 
######################################################################################################################################################################
### Fonction predict : 

@app.post("/predict")
def predict(user_input : UserInputPredict):
    """
    Effectue une prédiction en utilisant un modèle pré-entrainé sur un ensemble de données.

    Args:
        user_input (UserInputPredict): Instance de la classe UserInputPredict contenant les données d'entrée pour la prédiction.

    Returns:
        dict: Dictionnaire contenant les résultats de la prédiction.
            - 'prediction' (str): La prédiction résultante ('Accordé' ou 'Refusé').
            - 'certitude' (float): Le niveau de certitude associé à la prédiction (entre 0 et 1).
            - 'proba' (float): La probabilité associée à la prédiction (entre 0 et 1).
    """
    
    #  Récupération du dictionnaire des features : 
    data = user_input.dict()
    
    # Création du Dataframe : 
    X = pd.DataFrame([data['X']])

    # Calcul de la probabilité :
    proba = float(pipeline.predict_proba(X)[:,1][0])
    
    # Calcul de la prédiction et des distan©ce pour le calcul de la certitude vis à vis du seuil spécifique : 
    if proba > threshold: 
        prediction = 'Refusé'
        max_distance = 1 - threshold
    else: 
        prediction = 'Accordé'
        max_distance = threshold
            
    # Calcul de la certitude : 
    min_distance = 0
    distance = abs(proba-threshold)
    certitude = (distance - min_distance) / (max_distance - min_distance)
    
    # Création du dictionnaire de résultats :
    result ={
        'prediction': prediction,
        'certitude': certitude,
        'proba': proba,
        } 

    return result


######################################################################################################################################################################
### Fonction get_features_importance :

@app.get("/get_features_importance")
def get_features_importance():
    importance = pipeline.named_steps['model'].feature_importances_
    importance = importance.tolist()
    return {'Importance': importance}


######################################################################################################################################################################
### Fonction get_shap_values:

@app.post("/get_shap_values")
def get_shap_values(user_input : UserInputPredict):
    """
    Calcul des valeurs SHAP pour le dossier courant.

    Args:
        user_input (UserInputPredict): Une instance de la classe UserInputPredict contenant les données (variables et valeurs du dossier).

    Returns:
        dict: Un dictionnaire contenant les valeurs SHAP, les valeurs de base et les valeurs de données.
    """
    
    #  Récupération du dictionnaire des features : 
    data = user_input.dict()
    
    # Création du Dataframe : 
    X = pd.DataFrame([data['X']])
    
    # Création d'un explainer SHAP : 
    explainer = shap.Explainer(pipeline.named_steps['model'])

    # Standardisation de X : 
    X = pipeline.named_steps['scaler'].transform(X)
    
    # Création de shap_values sur X : 
    shap_values = explainer(X)
    
    # Récupération de shap_values pour la classe 1 : 
    shap_values_class_1 = shap_values.values[0, :, 1]
    
    # Récupération des bas_values pour la classe 1 : 
    base_value_class_1 = shap_values.base_values[0, 1]
    
    # Récupération des data : 
    data_values = shap_values.data[0]
    
    return {
        'shap_values': shap_values_class_1.tolist(),
        'base_values': base_value_class_1,
        'data_values': data_values.tolist(),
        }
