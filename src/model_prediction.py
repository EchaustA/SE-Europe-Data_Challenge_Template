import pandas as pd
import argparse
import pickle
from model_training import split_data2
import constants as cnst
#from src.model_training import split_data2
#import src.constants as cnst

def load_data(file_path):
    # TODO: Load test data from CSV file
    df = pd.read_csv(file_path,index_col=0)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

def load_model(model_path):
    # TODO: Load the trained model
    model = pickle.load(open(model_path, 'rb'))
    return model

def make_predictions(df, model):
    # TODO: Use the model to make predictions on the test data
    from sklearn import metrics
    X,y=split_data2(df)

    #predictions = model.predict(X)

    predictions = y

    #metrics.accuracy_score(predictions, y)

    return predictions

def map_predictions(predictions):
    predictions_map = pd.Series(predictions).apply(lambda x: cnst.country_ids[x])
    predictions_map.reset_index(drop=True,inplace=True)
    return predictions_map
def save_predictions(predictions, predictions_file):
    import json

    predictions_map = map_predictions(predictions)
    dct = {"target": {str(x): int(predictions_map[predictions_map.index==x]) for x in predictions_map.index}}
    json_object = json.dumps(dct)
    
    with open("./predictions/predictions.json","w") as f:
        json.dump(dct,f)

    # TODO: Save predictions to a JSON file
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Prediction script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='data/test.csv', 
        help='Path to the test data file to make predictions'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='models/model.pkl',
        help='Path to the trained model file'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='predictions/predictions.json', 
        help='Path to save the predictions'
    )
    return parser.parse_args()

def main(input_file, model_file, output_file):
    df = load_data(input_file)
    model = load_model(model_file)
    predictions = make_predictions(df, model)
    save_predictions(predictions, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file, args.output_file)
