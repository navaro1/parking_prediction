import pandas as pd
import numpy as np
import calendar

# Load and split data
raw_parking_data = pd.read_csv("./data/dataset.csv", parse_dates=[3])
# Znaleźć informacje o max i min w każdej kategorii oraz o ich rozmiarze
overview_data = pd.DataFrame()
for scn in pd.unique(raw_parking_data.SystemCodeNumber):
    scn_data = raw_parking_data[raw_parking_data['SystemCodeNumber'] == scn]
    overview_data = overview_data.append({'scn': scn,
                                          'min_date': min(scn_data.index),
                                          'max_date': max(scn_data.index),
                                          'length': len(scn_data)}, ignore_index=True)
overview_data = overview_data.set_index('scn')
# do wywalenia NIA North oraz BHMBRTARC01
parkings_to_delete = ['BHMBRTARC01', 'NIA North']
cleaned_parking_data = raw_parking_data[~raw_parking_data['SystemCodeNumber'].isin(parkings_to_delete)]

# Transform data
parking_data = pd.DataFrame()
parking_data['Occupied'] = cleaned_parking_data['Occupancy'] / cleaned_parking_data['Capacity']
parking_data['SystemCodeNumber'] = raw_parking_data['SystemCodeNumber']

overview_data = overview_data.drop(['BHMBRTARC01', 'NIA North'])
mean_length = overview_data['length'].mean()
test_data_size = int(mean_length * 0.2)


def to_cyclical_sin(value, n_period):
    return np.sin(value * (2. * np.pi / n_period))


def to_cyclical_cos(value, n_period):
    return np.cos(value * (2. * np.pi / n_period))


# funkcja sinus/cosinus na godzinę/dzień tygodnia
parking_data['SecondsSin'] = raw_parking_data['LastUpdated'].map(lambda idx: to_cyclical_sin(idx.second, 60))
parking_data['SecondsCos'] = raw_parking_data['LastUpdated'].map(lambda idx: to_cyclical_cos(idx.second, 60))

parking_data['MinSin'] = raw_parking_data['LastUpdated'].map(lambda idx: to_cyclical_sin(idx.minute, 60))
parking_data['MinCos'] = raw_parking_data['LastUpdated'].map(lambda idx: to_cyclical_cos(idx.minute, 60))

parking_data['HourSin'] = raw_parking_data['LastUpdated'].map(lambda idx: to_cyclical_sin(idx.hour, 24))
parking_data['HourCos'] = raw_parking_data['LastUpdated'].map(lambda idx: to_cyclical_cos(idx.hour, 24))

parking_data['WeekdaySin'] = raw_parking_data['LastUpdated'].map(lambda idx: to_cyclical_sin(idx.weekday(), 7))
parking_data['WeekdayCos'] = raw_parking_data['LastUpdated'].map(lambda idx: to_cyclical_cos(idx.weekday(), 7))

parking_data['DaysOfMonthSin'] = raw_parking_data['LastUpdated'].map(lambda idx:
                                                                     to_cyclical_sin(idx.day,
                                                                                     calendar.monthrange(idx.year,
                                                                                                         idx.month)[1]))
parking_data['DaysOfMonthCos'] = raw_parking_data['LastUpdated'].map(lambda idx:
                                                                     to_cyclical_cos(idx.day,
                                                                                     calendar.monthrange(idx.year,
                                                                                                         idx.month)[1]))

scn_to_label_map = {
    'BHMBCCMKT01': 'CityCentre',
    'BHMBCCPST01': 'CityCentre',
    'BHMBCCSNH01': 'Transport',
    'BHMBCCTHL01': 'CityCentre',
    'BHMBRCBRG01': 'CityCentre',
    'BHMBRCBRG02': 'Transport',
    'BHMBRCBRG03': 'CityCentre',
    'BHMBRTARC01': 'Hotel',
    'BHMEURBRD01': 'CityCentre',
    'BHMEURBRD02': 'CityCentre',
    'BHMMBMMBX01': 'ShoppingMall',
    'BHMNCPHST01': 'CityCentre',
    'BHMNCPLDH01': 'ShoppingMall',
    'BHMNCPNHS01': 'CityCentre',
    'BHMNCPNST01': 'ShoppingMall',
    'BHMNCPPLS01': 'Transport',
    'BHMNCPRAN01': 'Hotel',
    'Broad Street': 'ShoppingMall',
    'Bull Ring': 'ShoppingMall',
    'NIA Car Parks': 'Stadium',
    'NIA North': 'Stadium',
    'NIA South': 'Stadium',
    'Others-CCCPS105a': 'Other',
    'Others-CCCPS119a': 'Other',
    'Others-CCCPS133': 'Other',
    'Others-CCCPS135a': 'Other',
    'Others-CCCPS202': 'Other',
    'Others-CCCPS8': 'Other',
    'Others-CCCPS98': 'Other',
    'Shopping': 'ShoppingMall'
}
for v in set(scn_to_label_map.values()):
    parking_data[v] = parking_data['SystemCodeNumber'].map(lambda val: 1 if scn_to_label_map[val] == v else 0)

test_extrapolation_data = pd.DataFrame()
test_interpolation_data = pd.DataFrame()
np.random.seed(0)
for scn in overview_data.index:
    test_data = parking_data[parking_data['SystemCodeNumber'] == scn]
    interpolation_idxs = np.random.randint(0, len(test_data), test_intraploation_points)
    interpolation_data = test_data.iloc[interpolation_idxs]
    test_interpolation_data = test_interpolation_data.append(interpolation_data)
    extrapolation_data = test_data.tail(test_extrapolation_points)
    test_extrapolation_data = test_extrapolation_data.append(extrapolation_data)
    parking_data = parking_data.drop(extrapolation_data.index)
    parking_data = parking_data.drop(interpolation_data.index)

validation_data_size = int(len(parking_data) * 0.2)
validation_idxs = np.random.randint(0, len(parking_data) - 1, validation_data_size)
validation_data = parking_data.iloc[validation_idxs]

train_data = parking_data.drop(validation_data.index)

validation_data = validation_data.drop('SystemCodeNumber', axis=1)
train_data = train_data.drop('SystemCodeNumber', axis=1)
test_interpolation_data = test_interpolation_data.drop('SystemCodeNumber', axis=1)
test_extrapolation_data = test_extrapolation_data.drop('SystemCodeNumber', axis=1)
