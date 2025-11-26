import pandas as pd
from fbprophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import logging

#Postavljanje nivo logovanja na ERROR
logging.getLogger('prophet').setLevel(logging.ERROR)

#Učitavanje podataka
data = pd.read_csv("podacioriginal.csv", delimiter=";", decimal=",")
data.rename(columns={"Datum": "ds"}, inplace=True)

#Priprema podataka za FBProphet
mortalitet_data = data[["ds", "Stopa mortaliteta"]].rename(columns={"ds": "ds", "Stopa mortaliteta": "y"})
fertilitet_data = data[["ds", "Stopa fertiliteta"]].rename(columns={"ds": "ds", "Stopa fertiliteta": "y"})

#FBProphet model za mortalitet
mortalitet_model = Prophet()
mortalitet_model.fit(mortalitet_data)

#FBProphet model za fertilitet
fertilitet_model = Prophet()
fertilitet_model.fit(fertilitet_data)

#ARIMA model za mortalitet
mortalitet_arima_model = ARIMA(data["Stopa mortaliteta"], order=(2, 1, 0))
mortalitet_arima_model_fit = mortalitet_arima_model.fit()

#ARIMA model za fertilitet
fertilitet_arima_model = ARIMA(data["Stopa fertiliteta"], order=(2, 1, 0))
fertilitet_arima_model_fit = fertilitet_arima_model.fit()

#Funkcija za predviđanje
def predvidjanje(model, periods):
    future = model.make_future_dataframe(periods=periods, freq='Y')  # Podesite frekvenciju na godišnje
    forecast = model.predict(future)
    return forecast.tail(periods)[["ds", "yhat"]]

#Predviđanja za mortalitet za 5 i 10 godina
mortalitet_fbprophet_pred_5 = predvidjanje(mortalitet_model, 5)
mortalitet_fbprophet_pred_10 = predvidjanje(mortalitet_model, 10)
mortalitet_arima_pred_5 = mortalitet_arima_model_fit.forecast(steps=5).reset_index().rename(columns={"index": "ds", 0: "predicted_mean"})
mortalitet_arima_pred_10 = mortalitet_arima_model_fit.forecast(steps=10).reset_index().rename(columns={"index": "ds", 0: "predicted_mean"})

#Predviđanja za fertilitet za 5 i 10 godina
fertilitet_fbprophet_pred_5 = predvidjanje(fertilitet_model, 5)
fertilitet_fbprophet_pred_10 = predvidjanje(fertilitet_model, 10)
fertilitet_arima_pred_5 = fertilitet_arima_model_fit.forecast(steps=5).reset_index().rename(columns={"index": "ds", 0: "yhat"})
fertilitet_arima_pred_10 = fertilitet_arima_model_fit.forecast(steps=10).reset_index().rename(columns={"index": "ds", 0: "yhat"})

'''
#Crtanje grafikona
plt.figure(figsize=(12, 8))

#Plot za mortalitet
plt.subplot(2, 1, 1)
plt.plot(data["ds"], data["Stopa mortaliteta"], label="Stvarna stopa mortaliteta")
plt.plot(mortalitet_fbprophet_pred_5["ds"], mortalitet_fbprophet_pred_5["yhat"], label="FBProphet (5 godina)")
plt.plot(mortalitet_fbprophet_pred_10["ds"], mortalitet_fbprophet_pred_10["yhat"], label="FBProphet (10 godina)")
plt.plot(mortalitet_arima_pred_5["ds"], mortalitet_arima_pred_5["predicted_mean"], label="ARIMA (5 godina)")
plt.plot(mortalitet_arima_pred_10["ds"], mortalitet_arima_pred_10["predicted_mean"], label="ARIMA (10 godina)")

plt.title("Predviđanje stope mortaliteta")
plt.legend()

#Plot za fertilitet
plt.subplot(2, 1, 2)
plt.plot(data["ds"], data["Stopa fertiliteta"], label="Stvarna stopa fertiliteta")
plt.plot(fertilitet_fbprophet_pred_5["ds"], fertilitet_fbprophet_pred_5["yhat"], label="FBProphet (5 godina)")
plt.plot(fertilitet_fbprophet_pred_10["ds"], fertilitet_fbprophet_pred_10["yhat"], label="FBProphet (10 godina)")
plt.plot(fertilitet_arima_pred_5["ds"], fertilitet_arima_pred_5["yhat"], label="ARIMA (5 godina)")
plt.plot(fertilitet_arima_pred_10["ds"], fertilitet_arima_pred_10["yhat"], label="ARIMA (10

POKUSALE SMO DA NAM SE PRIKAZE GRAFIK ALI NIKAKO NECE DA RADI PA SMO TO IZOSTAVILE
'''

'''#Ispis predviđanja
print("\nPredviđanja za stopu mortaliteta (5 godina):\n", mortalitet_fbprophet_pred_5.to_string(index=False))
print("\nPredviđanja za stopu mortaliteta (10 godina):\n", mortalitet_fbprophet_pred_10.to_string(index=False))
print("\nPredviđanja za stopu fertiliteta (5 godina):\n", fertilitet_fbprophet_pred_5.to_string(index=False))
print("\nPredviđanja za stopu fertiliteta (10 godina):\n", fertilitet_fbprophet_pred_10.to_string(index=False))
'''

#BEZ REDNIH BTOJEVA
print('\n\n')
print("Predviđanja za stopu mortaliteta:")
print("FBProphet (5 godina):\n", mortalitet_fbprophet_pred_5.to_string(index=False))
print("FBProphet (10 godina):\n", mortalitet_fbprophet_pred_10.to_string(index=False))
print("ARIMA (5 godina):\n", mortalitet_arima_pred_5.to_string(index=False))
print("ARIMA (10 godina):\n", mortalitet_arima_pred_10.to_string(index=False))

print("\nPredviđanja za stopu fertiliteta:")
print("FBProphet (5 godina):\n", fertilitet_fbprophet_pred_5.to_string(index=False))
print("FBProphet (10 godina):\n", fertilitet_fbprophet_pred_10.to_string(index=False))
print("ARIMA (5 godina):\n", fertilitet_arima_pred_5.to_string(index=False))
print("ARIMA (10 godina):\n", fertilitet_arima_pred_10.to_string(index=False))