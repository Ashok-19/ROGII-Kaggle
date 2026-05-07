# Slide 1
Wellbore Geology Prediction

# Slide 2
Data Available
Zoomed in TVT plot
Each well has two CSV files:
Well1XXXX__horizontal_well.csv – horizontal well data
Well1XXXX__typewell__Typewell2XXXX.csv – Typewell (vertical well data)

# Slide 3
Horizontal Well Data
TVT (geology of the wellbore) must be predicted. This value is provided only for the training dataset
Top depth of each geological formation (provided only in the training dataset)
Provided for training and for prediction
Units: feet
MD – measured depth (well length)XYZ – coordinates of each point of the well
GR – GR – gamma ray values measured at each point of the horizontal well (some values may be NaN)
TVT – geology values (see next slide)
TVT_input – same geology values available until the Prediction Start (PS) point
Well10001__horizontal_well.csv file

# Slide 4
Vertical (Typewell) Well Data
Well10001__typewell__Typewell20001.csv file
Each horizontal well is assigned one typewellUnits: feet
TVT – true vertical thickness (depth in the vertical well)
GR – Gamma Ray Values measured at each point of the vertical well
Geology – name of geological layer
Typewell assigned to Horizontal well

# Slide 5
Goal: get TVT values from GR data
TVT value for typewell (vertical well) is always known
TVT values for the horizontal well are known until the Prediction Start (PS) point
The goal is to calculate TVT values beyond the PS point using XYZ and GR from the horizontal well and TVT and GR from the typewell

# Slide 6
Goal: get TVT values from GR data
Gamma Ray (GR)
GR projected on TVT track
GR signature matches Typewell GRTVT is increasing
TVT is decreasing
GR projected on TVT track
GR signature matches Typewell GRTVT is decreasing
TVT is increasing

# Slide 7
Goal: get TVT values from GR data
TVT incr
TVT decr
TVT constant
GR projected on TVT track
GR signature is constantTVT is constant

# Slide 8
Understanding TVT from Gamma Ray data
VIDEO: GR behavior helps us understand the TVT (geology) along each portion of the horizontal well
TVT scale

# Slide 9
Gamma ray in the horizontal well has better resolution
TVT scale
Well Path
Prediction Start (PS)
Gamma Ray (GR)
Typewell GR
GR before PS
GR after PS
GR from the vertical well has better resolution than GR from the typewellThe green GR correlates better with the red GR than with the typewell GR (black)
It may be better to use GR data from the horizontal well before the PS point, combined with deeper TVT data, to correlate the rest of the lateral

# Slide 10
Map view of all training and validation wells

# Slide 11
3D view of all training and validation wells

# Slide 12
Geology is almost flat
Geology is dipping
Geology is dipping in this direction
The azimuth of horizontal drilling affects the expected geology dip
The geology of an offset well can help predict the geology of the current well

# Slide 13
The geology of an offset well can help predict the geology of the current well
Geological dips behave similarly in neighboring wells
Dip
Dip
Flat
Flat

# Slide 14
How to evaluate Prediction quality
Each horizontal well has a TVT values that need to be predicted (one foot step)dTVT = manualTVT-predictedTVT for each point that is predictedPrediction quality is measured as the RMSE of all dTVT values
