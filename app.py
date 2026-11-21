
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("car_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    car_details = None
    error = None

    if request.method == "POST":

        try:
            
            brand = request.form["brand"]
            model_name = request.form["model"]

            year = int(request.form["year"])
            km_driven = int(request.form["km_driven"])

            fuel = request.form["fuel"]
            seller_type = request.form["seller_type"]
            transmission = request.form["transmission"]
            owner = request.form["owner"]

            mileage = float(request.form["mileage"])
            engine = int(request.form["engine"])
            power = float(request.form["power"])
            seats = int(request.form["seats"])

            if not 2008 <= year <= 2026:
                raise ValueError("Invalid manufacturing year")

            if km_driven < 0 or mileage <= 0:
                raise ValueError("Invalid kilometres or mileage")

            if engine <= 0 or power <= 0:
                raise ValueError("Invalid engine or power")

            if seats <= 0:
                raise ValueError("Invalid number of seats")

            input_data = pd.DataFrame([{
                "brand": brand,
                "model": model_name,
                "year": year,
                "km_driven": km_driven,
                "fuel": fuel,
                "seller_type": seller_type,
                "transmission": transmission,
                "owner": owner,
                "mileage_kmpl": mileage,
                "engine_cc": engine,
                "max_power_bhp": power,
                "seats": seats
            }])
           
            price_lakh = float(model.predict(input_data)[0])

            if not pd.notna(price_lakh):
                raise ValueError("Unable to predict price")

            price_lakh = max(0.01, price_lakh)

        
            price_rupees = price_lakh * 100000

            prediction = f"₹ {price_rupees:,.0f}"

            car_details = f"{year} {brand} {model_name}"

        except (ValueError, KeyError) as e:
            error = str(e) or "Please enter valid car details."

    return render_template(
        "index.html",
        prediction=prediction,
        car_details=car_details,
        error=error
    )
if __name__ == "__main__":
    app.run(debug=True)
