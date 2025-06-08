import matplotlib.pyplot as plt
import os
from flask import Flask, request, jsonify, send_from_directory, redirect
from dotenv import load_dotenv
import onnxruntime as ort
import matplotlib
import requests
import json
from typing import TypedDict, List

matplotlib.use('Agg')

load_dotenv()

# import tensorflow as tf
import numpy as np
from datetime import datetime, timedelta, UTC, timezone, date
import pandas as pd

WINDOW_SIZE = 30
INTO_FUTURE = 30
TOKEN="EWT"
TOKEN_NAME="Energy Web Token"
FILE_PATH = "./data/EWT-USD.json"
COINDESK_API_BASE_URL = "https://data-api.coindesk.com"
ENV = os.getenv("FLASK_ENV")
FRONTEND_DIR = os.path.abspath("./gui/dist")

is_dev = ENV == 'development'
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="/")
port = int(os.getenv("PORT", 5000))  # 5000 — порт за замовчуванням
model = ort.InferenceSession("./llm/model_1_dense.onnx")

# https://developers.coindesk.com/documentation/data-api/index_cc_v1_historical_days
def fetch_ewt_history():
  url = COINDESK_API_BASE_URL + "/index/cc/v1/historical/days"
  params={
    "market": "cadli",
    "instrument": TOKEN + "-USD",
    # "limit": 5000,
    "aggregate":1,
    "fill":"true",
    "apply_mapping":"true",
    "response_format": "JSON",
    "api_key": os.getenv("COINDESK_API_KEY")
  }
  headers={
    "Content-type":"application/json; charset=UTF-8"
  }
  response = requests.get(url=url, params=params, headers=headers)
  if response.status_code != 200:
    raise RuntimeError(f"❌ Запит не вдалось виконати: {response.status_code} - {response.text}")
  return response.json().get("Data", [])

def get_future_dates(start_date, into_future, offset=1):
  """
  Returns array of datetime values from ranging from start_date to start_date+horizon.

  start_date: date to start range (np.datetime64)
  into_future: number of days to add onto start date for range (int)
  offset: number of days to offset start_date by (default 1)
  """
  if isinstance(start_date, str):
    start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
  start_date = start_date + timedelta(offset) # specify start date, "D" stands for day
  end_date = start_date + timedelta(into_future) # specify end date
  return [d.strftime("%Y-%m-%d") for d in pd.date_range(start=start_date, end=end_date, freq="D")]

def make_future_forecast(values, model, into_future, window_size=WINDOW_SIZE) -> list:
  """
  Покроковий прогноз на основі ONNX-моделі через onnxruntime.

  :param values: список історичних значень (наприклад, цін)
  :param session: onnxruntime.InferenceSession
  :param into_future: скільки кроків передбачити
  :param window_size: розмір вікна (кількість останніх значень, які подаються на вхід)
  :return: список передбачених значень (float)
  """
  input_name = model.get_inputs()[0].name
  output_name = model.get_outputs()[0].name
  # 2. Make an empty list for future forecasts/prepare data to forecast on
  future_forecast = []
  last_window = values[-window_size:] # only want preds from the last window (this will get updated)

  # 3. Make INTO_FUTURE number of predictions, altering the data which gets predicted on each time
  for _ in range(into_future):

    # Predict on last window then append it again, again, again (model starts to make forecasts on its own forecasts)
    input_array = np.array([last_window], dtype=np.float32)  # shape: (1, window_size)
    output = model.run([output_name], {input_name: input_array})
    prediction = float(output[0][0][0])  # беремо перше число з прогнозу

    # Append predictions to future_forecast
    future_forecast.append(prediction)
    # print(prediction)

    # Update last window with new pred and get WINDOW_SIZE most recent preds (model was trained on WINDOW_SIZE windows)
    last_window = np.append(last_window, prediction)[-window_size:]

  return future_forecast

def plot_time_series(timesteps, values, format='.', start=0, end=None, label=None):
  """
  Plots a timesteps (a series of points in time) against values (a series of values across timesteps).

  Parameters
  ---------
  timesteps : array of timesteps
  values : array of values across time
  format : style of plot, default "."
  start : where to start the plot (setting a value will index from start of timesteps & values)
  end : where to end the plot (setting a value will index from end of timesteps & values)
  label : label to show on plot of values
  """
  if len(timesteps[start:end]) != len(values[start:end]):
    min_len = min(len(timesteps[start:end]), len(values[start:end]))
    timesteps = timesteps[start:end][:min_len]
    values = values[start:end][:min_len]
  else:
      timesteps = timesteps[start:end]
      values = values[start:end]
  # Plot the series
  plt.plot(timesteps[start:end], values[start:end], format, label=label)
  plt.title("Це заголовок графіка")  # додає заголовок
  plt.xlabel("Time")
  plt.ylabel(TOKEN + " Price")
  if label:
    plt.legend(fontsize=14) # make label bigger
  plt.grid(True)

# def get_latest_timestamp(data):
#   if not data:
#       return 0
#   return max(item["TIMESTAMP"] for item in data)

def get_missing_dates(existing_dates):
  if not existing_dates:
    return []

  last_date = max(existing_dates)
  today = datetime.now(UTC).date()

  expected_dates = {last_date + timedelta(days=i) for i in range(1, (today - last_date).days + 1)}
  missing = sorted(expected_dates - set(existing_dates))
  if missing:
    print("🟥 Missing dates:")
    for d in missing:
      print(d.isoformat())
  else:
    print("✅ No missing dates — data is up to date.")
  return missing

def save_data(file_path, data):
  with open(file_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

def filter_by_missing_dates(response_data: list, missing_dates: list[date]) -> list:
  # Convert missing_dates to a set of integer timestamps (midnight UTC)
  missing_ts_set = {
    int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())
    for d in missing_dates
  }

  # Filter the records
  filtered = [
    item for item in response_data
    if item.get("TIMESTAMP") in missing_ts_set
  ]

  return filtered

def load_existing_data(file_path):
  if not os.path.exists(file_path):
    return []
  with open(file_path, "r", encoding="utf-8") as f:
    return json.load(f)

def get_raw_data():
  data = load_existing_data(FILE_PATH)
  dates = {datetime.fromtimestamp(item["TIMESTAMP"], tz=timezone.utc).date() for item in data}
  existing_dates = sorted(dates)
  missing_dates = get_missing_dates(existing_dates)

  if len(missing_dates) == 0:
    print(f"✅ Усі дані оновлені. Нічого не потрібно робити.")
    return data

  new_data_list = fetch_ewt_history()
  new_data = filter_by_missing_dates(new_data_list, missing_dates)
  full_data = data + new_data
  full_data.sort(key=lambda x: x["TIMESTAMP"])
  save_data(FILE_PATH, full_data)
  print(f"✅ Додано {len(new_data)} нових записів")

  return full_data

class RawMarketData(TypedDict):
  UNIT: str
  TIMESTAMP: int
  TYPE: str
  MARKET: str
  INSTRUMENT: str
  OPEN: float
  HIGH: float
  LOW: float
  CLOSE: float
  FIRST_MESSAGE_TIMESTAMP: int
  LAST_MESSAGE_TIMESTAMP: int
  FIRST_MESSAGE_VALUE: float
  HIGH_MESSAGE_VALUE: float
  HIGH_MESSAGE_TIMESTAMP: int
  LOW_MESSAGE_VALUE: float
  LOW_MESSAGE_TIMESTAMP: int
  LAST_MESSAGE_VALUE: float
  TOTAL_INDEX_UPDATES: int
  VOLUME: float
  QUOTE_VOLUME: float
  VOLUME_TOP_TIER: float
  QUOTE_VOLUME_TOP_TIER: float
  VOLUME_DIRECT: float
  QUOTE_VOLUME_DIRECT: float
  VOLUME_TOP_TIER_DIRECT: float
  QUOTE_VOLUME_TOP_TIER_DIRECT: float

# class MarketData(TypedDict):
#   TIMESTAMP: int
#   CLOSE: float

def get_full_data(data: List[RawMarketData]):
  df = pd.DataFrame(data)
  df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], unit="s")
  df.set_index("TIMESTAMP", inplace=True)
  df.sort_index(inplace=True)
  chart = pd.DataFrame(df["CLOSE"]).rename(columns={"CLOSE": "Price"})
  start_date=chart.index[0].strftime('%Y-%m-%d')
  end_date=chart.index[-1].strftime('%Y-%m-%d')
  timesteps = chart.index.to_numpy()
  prices = chart["Price"].to_numpy()

  return {
    "chart": chart,
    "start_date": start_date,
    "end_date": end_date,
    "timesteps": timesteps,
    "prices": prices
  }

def get_chart(data):
  return [
    {"date": ts.strftime('%Y-%m-%d'), "price": float(price)}
    for ts, price in data["Price"].items()
  ]

def plot_chart(chart, prices, future_forecast, next_time_steps, max_date, max_price):
  plot_time_series(chart.index, prices, start=0, format="-", label="Actual Price")
  plot_time_series(next_time_steps, future_forecast, format="-", label="Predicted Price")
  start_date=next_time_steps[0]
  end_date=next_time_steps[-1]
  now=datetime.now(UTC)
  formatted = f"{max_price:.4f}"
  plot_name = "/static/images/ewt-predict_" + start_date.strftime("%Y-%m-%d") + "_" + end_date.strftime("%Y-%m-%d") + "_" + now.strftime("%Y-%m-%d") + ".png"
  plt.title("Найкраща ціна буде: " + formatted + " " + TOKEN + "/USD" + " (" + max_date.strftime("%Y-%m-%d") + ")" + " для періоду: " + start_date.strftime("%Y-%m-%d") + " / " + end_date.strftime("%Y-%m-%d") + " (" + str(len(future_forecast)) + "днів)")
  plt.savefig('.' + plot_name)
  return plot_name


@app.route("/")
def index():
  if os.path.exists(FRONTEND_DIR):
    return send_from_directory(FRONTEND_DIR, "index.html")
  return "Привіт із Flask!"

# http://127.0.0.1:3333/assets/images/ewt-predict-2025-06-07-2025-08-06.png
# Статичні файли
@app.route("/static/<path:filename>")
def static_files(filename):
  return send_from_directory("./static", filename)

@app.route("/test")
def test():
  return "hello world!"

@app.route("/api/chart")
def getChart():
  raw_data = get_raw_data()
  data = get_full_data(raw_data)
  return get_chart(data['chart'])

# http://127.0.0.1:3333/api/predict
# http://127.0.0.1:3333/api/predict?into_future=60
@app.route("/api/predict")
def predict():
  into_future = request.args.get("into_future", default=INTO_FUTURE, type=int)
  # window_size = request.args.get("window_size", default=WINDOW_SIZE, type=int) // TODO:  index: 1 Got: 60 Expected: 30 Please fix either the inputs/outputs or the model.
  if not (1 <= into_future <= 365):
    return jsonify({"error": "Параметр 'into_future' має бути в діапазоні 1–365"}), 400

  raw_data = get_raw_data()
  data = get_full_data(raw_data)
  future_forecast = make_future_forecast(values=data['prices'],
                                       model=model,
                                       into_future=into_future,
                                       window_size=WINDOW_SIZE)
  last_timestep = data['chart'].index[-1]
  next_time_steps = get_future_dates(start_date=last_timestep,
                                   into_future=into_future)
  plt.figure(figsize=(10, 7))
  next_time_steps = [pd.Timestamp(t).to_pydatetime() for t in next_time_steps]

  max_index = np.argmax(future_forecast)
  max_date = str(next_time_steps[max_index])
  max_price = float(future_forecast[max_index])
  future_data = [{"date": str(date), "price": float(price)} for date, price in zip(next_time_steps, future_forecast)]
  return {
    "max_date": max_date,
    "max_price": max_price,
    "actual_prices": get_chart(data['chart']),
    "predicted_prices": future_data,
    "plot": plot_chart(data['chart'], data['prices'], future_forecast, next_time_steps, next_time_steps[max_index], max_price)
  }

# http://127.0.0.1:3333/api/future-dates?into_future=1&start_date=2025-07-01
@app.route("/api/future-dates")
def futureDates():
  start_date = request.args.get("start_date", default=datetime.now(UTC).date())
  into_future = request.args.get("into_future", default=30, type=int)
  next_time_steps = get_future_dates(start_date=start_date, into_future=into_future)

  return {"result": {
     "start_date": str(start_date),
     "into_future": into_future,
     "next_time_steps": [str(d) for d in next_time_steps]
  }}

# Обробник 404 — повертає index.html у продакшн
@app.errorhandler(404)
def not_found(e):
  return redirect("/")

if __name__ == "__main__":
  app.run(debug=True, port=port)
