from flask import Flask, render_template, jsonify
from flask import request
import joblib
from tensorflow import keras

app = Flask(__name__)

@app.route("/")
def home():
    
    return render_template("index.html")

@app.route("/",methods=['POST'])
def getvalues():
    bed = request.form['bed']
    full_bath = request.form['full_bath']
    half_bath = request.form['half_bath']
    property_area = request.form['property_area']
    years_old = request.form['years_old']
    distance_downtown = request.form['distance_downtown']
    lot_size = request.form['lot_size']
    basement = request.form['basement']
    garage = request.form['garage']
    walk_score = request.form['walk_score']
    bike_score = request.form['bike_score']
    transit_score = request.form['transit_score']
    house = request.form['house']
    condo = request.form['condo']
    townhouse = request.form['townhouse']
    #print(bed)

    # Convert to numeric
    bed = int(bed)
    full_bath = int(full_bath)
    half_bath = int(half_bath)
    property_area = float(property_area)
    years_old = int(years_old)
    distance_downtown = float(distance_downtown)
    lot_size = float(lot_size)
    basement = int(basement)
    garage = int(garage)
    walk_score = int(walk_score)
    bike_score = int(bike_score)
    transit_score = int(transit_score)
    house = int(house)
    condo = int(condo)
    townhouse = int(townhouse)

    # Testing ML Model
    filename = '/data/LogisticRegression.sav'

    joblib_LR_model = joblib.load(filename)
    joblib_LR_model

    test_data = [[bed, full_bath, half_bath, property_area, years_old, distance_downtown, lot_size, basement, garage, walk_score, bike_score, transit_score, house, condo, townhouse]]

    Ypredict = joblib_LR_model.predict(test_data)

    #reconstructed_model = keras.models.load_model("data/my_h5_model.h5")

    #score = reconstructed_model.fit(test_data)

    return render_template("pass.html", Ypredict=Ypredict)

if __name__ == "__main__":
    app.run(debug=True)