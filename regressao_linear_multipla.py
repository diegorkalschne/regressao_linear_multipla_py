from dateutil.parser import parserinfo
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

k = int(input("Número de variáveis independentes ")) + 1 # Número de variaveis independentes
n = int(input("Número de dados de cada variável "))
variable_values_table = pd.DataFrame({})


inputs = []
print("Insira os números separados por vírgula")
for i in range(k):
  if (k - 1) == i:
    line = input("y ")
    strings = line.split(',')
    x = [int(s) for s in strings]
  else:
    line = input("x{} ".format(i + 1))
    strings = line.split(',')
    x = [int(s) for s in strings]

  inputs.append(x)



for i in range(1,k + 1):
  if i == k:
    variable_values_table["y"] = 1
    break
  variable_values_table["x{}".format(i)] = 1


for i,e in enumerate(variable_values_table.columns):
  variable_values_table[e] = inputs[i]


numpy_matriz_values = variable_values_table.to_numpy()
for columnIndex,columnName in enumerate(variable_values_table.columns):
  if(columnName != 'y'):
    variable_values_table["{}{}".format(columnName,'^2')] = variable_values_table[columnName] * variable_values_table[columnName]

for columnIndex,columnName in enumerate(variable_values_table.columns):
  count = 1
  for i in range(columnIndex + 1,k):
      variable_values_table["{}{}".format(columnName,variable_values_table.columns[columnIndex+count])] = np.multiply(numpy_matriz_values[:,columnIndex] , numpy_matriz_values[:,i] )
      count += 1


independent_variables = variable_values_table.iloc[:, :variable_values_table.columns.get_loc("y")]
dependent_variables = variable_values_table["y"]
X = np.c_[np.ones(len(independent_variables)), independent_variables[list(independent_variables.columns.values)]]

columns = list(independent_variables.columns.values)
columns = np.array(['b' + str(i) for i in range(len(columns))]) # cria o novo array com uma
columns = np.append(columns,"b{}".format(len(independent_variables.columns.values)))

none_array = np.empty(len(columns))
values = [{x: none_array} for x in columns]

independent_variable_sums = []

independent_variable_quantity = len(independent_variables.columns)

A = pd.DataFrame(values,columns=columns)
B = pd.DataFrame(np.empty(independent_variable_quantity + 1), columns=['y'])

for index,independent_variable in enumerate(independent_variables):
  array_variable = independent_variables[independent_variable]
  independent_variable_sums.append(array_variable)
  none_array = np.empty(independent_variable_quantity)
  b_array = np.empty(k, dtype=object)
  b_array[0] = array_variable
  b_array[1:] = None
  A["b{}".format(index + 1)] = b_array


A['b0'] = [n] + independent_variable_sums

length = A.shape[0]
for row_index in range(1,length):
  for column_index in range(1,length):
    A.values[row_index][column_index] = np.multiply(A.values[0][column_index] ,A.values[row_index][0]).sum()


B.values[0][0] = dependent_variables.values.sum()

for i in range(1,length):
  B.values[i][0] = np.multiply( dependent_variables.values, A.values[i][0]).sum()


for i in range(1,length):
  A.values[0][i] = A.values[0][i].sum()
  A.values[i][0] = A.values[i][0].sum()


ones = np.ones(n)
independent_variables.insert(0,'x', ones)

A_t = independent_variables.T
teste = np.dot(A_t, independent_variables)
t = np.linalg.inv(teste)
print('')
print(A)
print(B)
print('')
C = np.dot(t,B)

index = 0
for i in C:
   print("b{} = {}".format(index,i[0]))
   index = 1 + index

print('')

x1 = np.array(inputs[0])
x2 = np.array(inputs[1])
y = np.array(inputs[len(inputs) - 1])
X = np.column_stack((x1, x2))
y = y.reshape(-1, 1)

regr = LinearRegression()
regr.fit(X, y)

x1_range = np.arange(x1.min(), x1.max())
x2_range = np.arange(x2.min(), x2.max())
X1, X2 = np.meshgrid(x1_range, x2_range)
Y = regr.predict(np.c_[X1.ravel(), X2.ravel()]).reshape(X1.shape)

fig = go.Figure()
fig.add_trace(go.Scatter3d(x=x1, y=x2, z=y.ravel(), mode='markers', marker=dict(size=4, color='blue'), name="3D plot"))
fig.add_trace(go.Surface(x=x1_range, y=x2_range, z=Y, colorscale='Viridis', opacity=0.5))

table_graph = go.Table(
    name="Table",
    header=dict(values= variable_values_table.columns,
    font=dict(size=10),
    align='left'),
    cells=dict(values=variable_values_table.transpose().values.tolist(),
    fill_color='lavender',
    align='left'),
    domain=dict(x=[0.5,1], y =[0,1]))

fig.add_trace(table_graph)
fig.update_layout(title='Regressão Linear Múltipla', scene=dict(xaxis_title='x1', yaxis_title='x2', zaxis_title='y',domain=dict(x=[0, 0.5], y=[0, 1])))
fig.show()