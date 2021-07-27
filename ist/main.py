
import tkinter as tk
import tkinter.font as tkFont
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from yahoo_fin.stock_info import get_data
from datetime import datetime
from keras.models import load_model

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class App:
    def __init__(self, master):

        frame = tk.Frame(master)

        #Setting dataset
        ticker_list = ["amzn", "eth-usd", "ba", "^gspc", "nflx", "fb", "tsla", "ko", "nsrgy", "pep", "mnst", "wmt", "t", "vz", "gm", "btc-usd", "mcd", "qsr", "msft", "sne"]
        historical_datas = {}
        for ticker in ticker_list:
            historical_datas[ticker] = get_data(ticker, index_as_date=True, interval="1d")

        self.df = historical_datas
        self.prediction_range = .7

        self.title_style = tkFont.Font(family="Arial", size=24)
        self.label_title = tk.Label(frame, text="Stock Prediction App", font=self.title_style)
        self.label_title.pack()

        self.frame_search = tk.Frame(frame, height=40)
        #self.frame_space = tk.Frame(frame)
        self.frame_fig = tk.Frame(frame)

        self.stock_selected = tk.StringVar(self.frame_search)
        self.stock_selected.set("tsla") # default value

        self.label_metrics = tk.Label(frame, text="")


        #Define Figure
        fig = Figure(figsize=(9, 4))
        self.ax = fig.add_subplot(111)
        self.ax.locator_params(axis="x", nbins=5)
        self.canvas = FigureCanvasTkAgg(fig,master=self.frame_fig)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(anchor=tk.NW)


        self.w = tk.OptionMenu(self.frame_search, self.stock_selected, "amzn", "eth-usd", "ba", "^gspc", "nflx", "fb",
                               "tsla", "ko", "nsrgy", "pep", "mnst", "wmt", "t", "vz", "gm", "btc-usd", "mcd",
                               "qsr", "msft", "sne", command=self.make_pred)
        self.w.pack(side=tk.RIGHT, fill=tk.BOTH, expand=tk.TRUE)
        self.label_m = tk.Label(self.frame_search, text="Stock", width=10)
        self.label_m.pack(side=tk.RIGHT)
        self.label_space = tk.Label(self.frame_search, text="", width=5)
        self.label_space.pack(side=tk.RIGHT)

        self.time_variable = tk.StringVar(self.frame_search)
        self.time_variable.set("Start") # default value
        self.w = tk.OptionMenu(self.frame_search, self.time_variable, "Start", "10Y", "5Y", command=self.make_pred)
        self.w.pack(side=tk.RIGHT, fill=tk.BOTH, expand=tk.TRUE)
        self.label_m = tk.Label(self.frame_search, text="Range", width=10)
        self.label_m.pack(side=tk.RIGHT)

        self.make_pred(self)

        #self.frame_space.pack(fill=tk.X, padx=5, pady=5);
        self.frame_search.pack(fill=tk.X, padx=5);
        self.label_metrics.pack()
        self.frame_fig.pack(side=tk.LEFT, padx=5, pady=3);

        frame.pack()

    @staticmethod
    def datetime_to_timestamp(x):
            return datetime.strptime(x.strftime('%Y%m%d'), '%Y%m%d')

    #Change make prediction on selected options
    def make_pred(self, var):
        data = self.df[self.stock_selected.get()]
        data.index = data.index.to_series().apply(self.datetime_to_timestamp)

        if self.time_variable.get() == "Start":
            data = self.df[self.stock_selected.get()]
        elif self.time_variable.get() == "10Y":
            data = data.loc["2010-01-01":]
        elif self.time_variable.get() == "5Y":
            data = data.loc["2015-01-01":]


        new_dataset = pd.DataFrame(index=range(0,len(data)),columns=['date','adjclose'])

        for i in range(0,len(data)):
            new_dataset['date'][i] = data.index[i]
            new_dataset['adjclose'][i]=data["adjclose"][i]

        final_set = new_dataset.values

        train_test_split = int(len(final_set)*self.prediction_range)
        train_data = final_set[:train_test_split]
        self.valid_data = final_set[train_test_split:]

        new_dataset.index = new_dataset.date
        new_dataset.drop('date', axis=1, inplace=True)

        final_set = new_dataset.values

        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(final_set)

        inputs_data = new_dataset[len(new_dataset)-len(self.valid_data)-60:].values
        inputs_data = inputs_data.reshape(-1, 1)
        inputs_data = scaler.transform(inputs_data)

        model = load_model("lstm_model.h5") #Copy of new_model.h5

        X_data=[]
        for i in range(60,inputs_data.shape[0]):
            X_data.append(inputs_data[i-60:i,0])
        X_data = np.array(X_data)
        X_data = np.reshape(X_data,(X_data.shape[0],X_data.shape[1],1))
        predicted_closing_price = model.predict(X_data)

        test_performance = predicted_closing_price
        predicted_closing_price = scaler.inverse_transform(predicted_closing_price)

        train_data=new_dataset[:train_test_split]
        self.valid_data=new_dataset[train_test_split:]
        self.valid_data['Predictions']=predicted_closing_price
        self.ax.clear()
        self.ax.plot(train_data["adjclose"], label='Close Data')
        self.ax.plot(self.valid_data[['adjclose',"Predictions"]], label='Close Data Prediction')
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Price - USD")
        self.ax.legend(['Adjusted Close Price', 'Adjusted Close Price', 'Prediction'])
        self.canvas.draw()

        self.get_sucess()

    # Update success metrics
    def get_sucess(self):

        mae = mean_absolute_error(self.valid_data["adjclose"], self.valid_data["Predictions"])
        mse = mean_squared_error(self.valid_data["adjclose"], self.valid_data["Predictions"])
        rmse = np.sqrt(mse)
        r2 = r2_score(self.valid_data["adjclose"], self.valid_data["Predictions"])

        self.label_metrics["text"] = "Prediction Accuracy: {:.2f}%     MSE: {:.2f}".format(r2*100, mse)


root = tk.Tk()
app = App(root)
root.mainloop()