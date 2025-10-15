# paste your project's model id below:
modelID = '6905cfd7-25eb-40f9-85ef-7c0373f9f715' # this id is from a demo project (stock prediction)

baseURL = 'https://aicode101.com'
api = {
    'test': '%s/api/model/%s/test' %
            (baseURL, modelID),
}

# show api
api

# import libraries
import requests
import numpy

# get the test data
httpResponse = requests.get(api['test'])
data = httpResponse.json()

# see the features that you're going to use
feature_names = data['features']
feature_names

# see the target that you're predicting
target_name = data['target']
target_name

# configure tools to plot pretty figures
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plot
plot.rcParams['axes.labelsize'] = 14
plot.rcParams['xtick.labelsize'] = 12
plot.rcParams['ytick.labelsize'] = 12

# plot the prediction v.s. answer
# (should be a straight line, if it is accurate)
predictions = data['predictions']
answers = data['answers']
plot.scatter(answers, predictions)

# group values to its corresponding feature
x = {}
for idx in range(len(answers)):
  for jdx, feature_name in enumerate(feature_names):
    if feature_name not in x:
      x[feature_name] = [data['tests'][idx][jdx]]
    else:
      x[feature_name] += [data['tests'][idx][jdx]]

# plot the features v.s. prediction
# (these plots show which features are more relevent to the predictions)
# (if it is scattered, then it is likely less relevent to the prediction)
for feature_name, content in x.items():
  fig = plot.figure()
  plot.title(feature_name + ' v.s. ' + target_name + '_prediction')
  plot.scatter(content, predictions)
