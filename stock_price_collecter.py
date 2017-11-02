import pandas_datareader.data as web
import pandas as pd
import datetime
import numpy as np


training_start_time = datetime.datetime(2013, 1, 1)
training_end_time = datetime.datetime(2015, 1, 1)
validation_start_time = training_end_time
validation_end_time = datetime.datetime.today()
COMPANIES = 20


# NYSE
url_nyse = "http://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nyse&render=download"
# Nasdaq
url_nasdaq = "http://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download"
# AMEX
url_amex = "http://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=amex&render=download"

def reshape_data(ary:list):
    refined_data = []
    for i in range(len(ary[0])):  # for each time
        prices_at_given_time = []
        print("i = %s"%i)
        for j in range(len(ary)):  # for each company
            prices_at_given_time.append(ary[j][i])
            print("j = %s"%j)
        refined_data.append(prices_at_given_time)

    return refined_data


df = pd.DataFrame.from_csv(url_nyse)
nyse_companies = df.index.tolist() # list of companies in NYSE

training_nyse_open = []
validation_nyse_open = []
counter = 0
train_time_length = 0
validation_time_length = 0
fail_counter = 0

while counter < COMPANIES:
    try:
        if(fail_counter > 5):
            counter += 1
            COMPANIES += 1
        company = nyse_companies[counter]
        training_open_price = web.DataReader(company, "yahoo", training_start_time, training_end_time)['Open']
        validation_open_price = web.DataReader(company, "yahoo", validation_start_time, validation_end_time)['Open']
        print("retrieved : " + company )

        if len(training_open_price) >= train_time_length:
            train_time_length = len(training_open_price)
        else:
            raise Exception

        if len(validation_open_price) >= validation_time_length:
            validation_time_length = len(validation_open_price)
        else:
            raise Exception

        if None in training_open_price or None in validation_open_price:
            raise Exception

        training_nyse_open.append(training_open_price)
        validation_nyse_open.append(validation_open_price)
        counter += 1
        fail_counter = 0

    except Exception as e:
        print("failed to get : " + company)
        fail_counter += 1
        # counter += 1
        # COMPANIES +=1


#training_nyse_open = np.load("train_hoge.npy")
#validation_nyse_open = np.load("validaiton_hoge.npy")

for i in range(len(training_nyse_open)):
    print(len(training_nyse_open[i]))

training_nyse_open = reshape_data(training_nyse_open)
validation_nyse_open = reshape_data(validation_nyse_open)

training_nyse_open = np.array(training_nyse_open)
validation_nyse_open = np.array(validation_nyse_open)


np.save("training_nyse_stocks.npy", training_nyse_open)
np.save("validation_nyse_stocks.npy", validation_nyse_open)

print("save complete")





#apple = web.DataReader("AAPL", "yahoo", start_time, end_time) # get apple stock data
# print(apple)

#apple_open = apple['Open'] # get only the opening price
# print(apple_open)



# output = []
# for i in range(len(apple_open)):
#     output.append(apple_open[i])
#
# output = np.array(output)
# print(output)
#
# np.save('apple_stocks.npy', output)



