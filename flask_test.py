from flask import Flask, render_template, request
import json
import pandas as pd
from keras.models import load_model
from joblib import dump, load


app = Flask(__name__,template_folder='template',static_folder='.', static_url_path='')
# app = Flask(__name__)

@app.route("/hello/<input>")
def hello(input): 
  return input

### The prediction API to output the forcasted the cat's lifespan based on the input data
@app.route("/cat/prediction", methods=['POST'])
def predict_age():
    api_field_type_map ={
        'breed': str,
        'last_vet_visit': str,
        'hair_length': float,
        'height': float,
        'num_vet_visit': float,
        'weight': float
    }
    
    api_from_values = {} 
    
    for api_field_name,api_field_type in api_field_type_map.items():
        api_from_values[api_field_name] = request.form.get(
            api_field_name,type=api_field_type
        )
    
    
    ## data process
    OneRecord = pd.DataFrame(api_from_values, index=[0])
    
    # from the input data, we won't have age, this is just a placeholder to satisfies column transformer
    # TODO: update column transformer only process input features
    OneRecord['age'] = 1.0
    OneRecord = OneRecord[['age','breed','hair_length','height','num_vet_visit','weight','last_vet_visit']]
    
    
    OneRecord = OneRecord.drop(columns=['last_vet_visit'])
    
    print(OneRecord)
    
    pipeline = load('keras_model_data_trans_pipeline.joblib')
    
    OneRecord = pipeline.transform(OneRecord)
    
    ##prediction
    model = load_model('cat_age_prediction_keras.h5')
    result = model.predict(OneRecord[:,:-1])
#     print(str(result))

    

    ## TODO: certainly there must be a better way to do this    
    return json.dumps(float(result[0][0]))

### The input form for the car lifespan prediction
@app.route("/cat/input")
def input_for_predict():
    form_config = [
        {'field':'breed', 'label':'Cat Breed'},
        {'field':'hair_length', 'label':'Cat hair length(in cm)'},
        {'field':'height', 'label':'Cat Height(in cm)'},
        {'field':'num_vet_visit', 'label':'How many times cats has visited vet after born'},
        {'field':'weight', 'label':'Cat Weight(in kg)'},
        {'field':'last_vet_visit', 'label':'Last time cat visit vet (optional)'},
    ]
    
    return render_template('cat_life_predict.html', form_config=form_config)


if __name__ == "__main__": app.run(debug=True,host='0.0.0.0')