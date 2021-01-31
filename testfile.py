## concat test files

import pandas as pd
import glob
import os

input_file = 'test_1-11'

file_list = glob.glob(os.path.join(input_file, '*.csv'))

test = []

for file in file_list:
    df = pd.read_csv(file, usecols=['value'])
    test.append(df)

testCombine = pd.concat(test, axis=1, ignore_index=True)
testCombine = testCombine.transpose()

# print(testCombine)

testCombine.to_csv('testCombine.csv')
