# Mottled
Two Dimensional Pattern Recognition with Prediction

Mottled combs through a sequence of real valued data looking for unique and similar relative patterns that are 2 dimensional. It rates patterns based on the number of occurances that a given pattern exists in the data (and other customizable factors). It doesn't look for absolute patterns in the data, rather the difference from a row to the previous row.

Using the input file provided (allForexData.csv) and the base parameters provided, it achieves roughly 43% Pass/Fail accuracy at predicting the next line of real valued data. Numeric accuracy is around 99.99%.

allForexData.csv is a collection of hourly Forex Data for AUDUSD commodity. Mottled can read multiple columns and rows of data and predict with wider pattern sizes. Other Forex data has been included (without labels, sorry)

MottledTuning.ods is a spreadsheet of (some) of the tuning I've done to attempt to improve accuracy.
