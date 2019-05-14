import pandas as pd
import numpy as np
import calendar

class ParkingDataLoader:
  def __init__(self, scn_to_categories=None, dataset_location="./data/dataset.csv"):
    self.__scn_to_categories = scn_to_categories if scn_to_categories is not None else __get_scn_to_labels()
    self.__dataset = pd.read_csv(dataset_location, parse_dates=[3])
    self.__data_overview = None
    
  def get_data_overview(self):
    return self.__data_overview if self.__data_overview is not None else self.__do_get_data_overview()
  
  def get_train_validation_test_datasets(validation_split=0.16, test_split=0.2, without_scns=['BHMBRTARC01', 'NIA North']):
    cleaned_parking_data = self.__dataset[~self.__dataset['SystemCodeNumber'].isin(without_scns)]
    
    pass
  
  def __do_get_data_overview(self):
    overview_data = pd.DataFrame()
    for scn in pd.unique(self.__dataset.SystemCodeNumber):
      scn_data = self.__dataset[self.__dataset['SystemCodeNumber'] == scn]
      overview_data = overview_data.append({'scn': scn,
                                              'min_date': min(scn_data['LastUpdated']),
                                              'max_date': max(scn_data['LastUpdated']),
                                              'length': len(scn_data)}, ignore_index=True)
    overview_data = overview_data.set_index('scn')
    self.__overview_data = overview_data
    return overview_data
  
  def to_cyclical_sin(value, n_period):
    return np.sin(value * (2. * np.pi / n_period))


  def to_cyclical_cos(value, n_period):
    return np.cos(value * (2. * np.pi / n_period))
    
  def __get_default_scn_to_labels():
    return {
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

def get_scn_to_labels():
  return {
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


# Load and split data
raw_parking_data = pd.read_csv("./data/dataset.csv", parse_dates=[3])
# Znaleźć informacje o max i min w każdej kategorii oraz o ich rozmiarze
overview_data = pd.DataFrame()
for scn in pd.unique(raw_parking_data.SystemCodeNumber):
    scn_data = raw_parking_data[raw_parking_data['SystemCodeNumber'] == scn]
    overview_data = overview_data.append({'scn': scn,
                                          'min_date': min(scn_data['LastUpdated']),
                                          'max_date': max(scn_data['LastUpdated']),
                                          'length': len(scn_data)}, ignore_index=True)
overview_data = overview_data.set_index('scn')
# do wywalenia NIA North oraz BHMBRTARC01
parkings_to_delete = ['BHMBRTARC01', 'NIA North']
cleaned_parking_data = raw_parking_data[~raw_parking_data['SystemCodeNumber'].isin(parkings_to_delete)]

# Transform data
parking_data = pd.DataFrame()
parking_data['Occupied'] = cleaned_parking_data['Occupancy'] / cleaned_parking_data['Capacity']
parking_data['SystemCodeNumber'] = raw_parking_data['SystemCodeNumber']


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

parking_data['DaysOfMonthSin'] = raw_parking_data['LastUpdated'].map(lambda idx: to_cyclical_sin(idx.day, calendar.monthrange(idx.year, idx.month)[1]))
parking_data['DaysOfMonthCos'] = raw_parking_data['LastUpdated'].map(lambda idx: to_cyclical_cos(idx.day, calendar.monthrange(idx.year, idx.month)[1]))

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
parking_data = parking_data.drop('SystemCodeNumber', axis=1)

def split_to_train_validation_test_datasets(data, test_split=0.2, validation_split=0.16):
  test_data_size = int(len(data) * test_split)
  test_idxs = np.random.randint(0, len(data) - 1, test_data_size)
  test_data = data.iloc[test_idxs]
  training_data = data.drop(test_data.index)
  validation_data_size = int(len(data) * validation_split)
  validation_idxs = np.random.randint(0, len(training_data) - 1, validation_data_size)
  validation_data = training_data.iloc[validation_idxs]
  training_data = training_data.drop(validation_data.index)
  return training_data, validation_data, test_data

training_data, validation_data, test_data = split_to_train_validation_test_datasets(parking_data)
