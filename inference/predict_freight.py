import joblib
import pandas as pd

MODEL_PATH = "models/predict_freight_model.pkl"

def load_model(model_path: str = MODEL_PATH):
    """
    Load trained freight cost prediction model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
        return model

def predict_freight_cost(input_data):
    """
    Predict freight cost for new vendor invoices.

    Parameters
    ----------
    input_data: dict
     
     Returns
     -------
     pd.DataFrame with predicted freight cost
     """
    model = load_model() 
    input_df = pd.DataFrame(input_data)

       # ✅ Ensure correct column names + order
    input_df = input_df[['Quantity', 'Dollars']]
    
    input_df['Predicted_Freight'] = model.predict(input_df).round()
    return input_df

if __name__ == "__main__":
    sample_data = {
        "quantity": [1200],
        "Dollars": [15000.0]
    }

    prediction = predict_freight_cost(sample_data)
    print(prediction)
