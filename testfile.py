""" stock_prediction / testfile.py :  """

"""

"""
__author__ = "Shuntaro Katsuda"

# encoding: utf-8


import datetime
from data_collector import collect_price

hoge = collect_price('TWTR', datetime.date(2017,1,1), datetime.date.today())

print(hoge)

