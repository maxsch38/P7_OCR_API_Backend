######################################################################################################################################################################
### Fichier de tests unitaire des Endpoints de backend.py
######################################################################################################################################################################

######################################################################################################################################################################
### Importation des librairies : 
import pytest
import random
import os 
import sys
from fastapi.testclient import TestClient

# Ajout du dossier parent dans la recherche de module : 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# Importation depuis backend.py : 
from backend import app, pipeline

######################################################################################################################################################################
# Décalaration du client : 
client = TestClient(app)

# Récupération de la liste des features : 
liste_features = pipeline.named_steps['scaler'].get_feature_names_out()

# Fixture pour input_data : 
@pytest.fixture
def input_data():
    """
    Fixture pour générer un vecteur client a tester. 
    """
    return {"X": {feature: random.uniform(0, 1) for feature in liste_features}}

######################################################################################################################################################################
# Fonctions de tests unitaires : 

def test_get_threshold():
    """
    Test du Endpoint /get_threshold
    """
    response = client.get("/get_threshold")
    assert response.status_code == 200
    assert "Threshold" in response.json()

def test_get_features_importance():
    """
    Test du Endpoint /get_features_importance
    """
    response = client.get("/get_features_importance")
    assert response.status_code == 200
    assert "Importance" in response.json()


@pytest.mark.filterwarnings("ignore:is_sparse is deprecated")
def test_predict(input_data):
    """
    Test du Endpoint /predict
    """
    response = client.post("/predict", json=input_data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "certitude" in response.json()
    assert "proba" in response.json()

@pytest.mark.filterwarnings("ignore:is_sparse is deprecated")
def test_get_shap_values(input_data):
    """
    Test du Endpoint /get_shap_values
    """
    response = client.post("/get_shap_values", json=input_data)
    assert response.status_code == 200
    assert "shap_values" in response.json()
    assert "base_values" in response.json()
    assert "data_values" in response.json()
    