""" stock_prediction / data_collector.py :  """

"""
collects data for learning
"""
__author__ = "Shuntaro Katsuda"

# encoding: utf-8

import datetime
import json
import pickle as pkl
import numpy as np
import pandas_datareader as web
import pandas as pd

training_start_time = datetime.datetime(2012, 1, 1)
training_end_time = datetime.datetime(2016, 1, 1)
validation_start_time = training_end_time
validation_end_time = datetime.datetime(2017, 6, 30)

def getCorpWithAttr(data: dict, attr: str = None, value: str = None, length: int = 10000):
    attr_dict = data[attr]
    output = []
    if not(attr == None or value == None):
        for k, v in attr_dict.items():
            if v == value:
                output.append(k)
    else:
        print(" is 2")
    return output

def collect_price(symbol: str, start_time: datetime.date, end_time: datetime.date):
    return web.DataReader(symbol, "google", start_time, end_time)['Open']

def dataframeToList(data: pd.DataFrame):
    return data.values.tolist()

def reshape_data(ary:list):
    """
    reshapes array and converts dataframe into 2 dimensional array
    :param ary:
    :return reshaped array:
    """
    print(ary)
    refined_data = []
    for i in range(len(ary[0])):  # for each time
        prices_at_given_time = []
        for j in range(len(ary)):  # for each company
            prices_at_given_time.append(ary[j][i])
            print("j = %s"%j)
        refined_data.append(prices_at_given_time)
    return refined_data

if __name__ == "__main__":

    # NYSE
    url_nyse = "http://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nyse&render=download"
    # Nasdaq
    url_nasdaq = "http://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download"
    # AMEX
    url_amex = "http://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=amex&render=download"

    nyse_companies_json = pd.DataFrame.from_csv(url_nyse).to_json()
    nyse_companies = json.loads(nyse_companies_json)

    # tech corps in list
    tech_corps = getCorpWithAttr(nyse_companies, "Sector", "Technology")
    print("found %s corps"%str(len(tech_corps)))

    tech_corps = tech_corps[:5]
    train = [None] * len(tech_corps)
    validation = [None] * len(tech_corps)

    count = 1
    for corp in tech_corps:
        fail_counter = 0
        print("corp[" + str(count) + "]: %s"%corp)
        while True:
            try:
                train.append(collect_price(corp, training_start_time, training_end_time))
                validation.append(collect_price(corp, validation_start_time, validation_end_time))
            except Exception as e:
                print(e)
                if fail_counter >= 5:
                    break
                print("failed... trying again")
                fail_counter += 1
        count += 1
    print("finished collecting all")

    # reshape data
    train = reshape_data(train)
    validation = reshape_data(validation)

    train = np.array(train)
    validation = np.array(validation)

    # save data
    np.save("data/training_tech_corps.npy", train)
    np.save("data/validation_tech_corps.npy", validation)

    print("--- train shape ---")
    print(train.shape)
