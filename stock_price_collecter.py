import pandas_datareader.data as web
import pandas as pd
import datetime
import numpy as np

start_time = datetime.datetime(2015, 1, 1)
end_time = datetime.datetime.today()
COMPANIES = 20


# NYSE
url_nyse = "http://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nyse&render=download"
# Nasdaq
url_nasdaq = "http://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download"
# AMEX
url_amex = "http://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=amex&render=download"


df = pd.DataFrame.from_csv(url_nyse)
nyse_companies = df.index.tolist() # list of companies in NYSE

nyse_open = []
counter = 0
while counter < 20:
    try:
        company = nyse_companies[counter]
        open_price = web.DataReader(company, "yahoo", start_time, end_time)['Open']
        print("retrieved : " + company )
        nyse_open.append(open_price)
        counter += 1
    except Exception as e:
        print("failed to get : " + company)

#print(nyse_open[1])

refined_data = []
for i in range(len(nyse_open[0])): # for each time
    prices_at_given_time = []
    for j in range(len(nyse_open)): # for each company
        prices_at_given_time.append(nyse_open[j][i])
    refined_data.append(prices_at_given_time)

data = np.array(refined_data)

print(data.shape)
# print(data)

np.save("nyse_stocks.npy", data)





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



