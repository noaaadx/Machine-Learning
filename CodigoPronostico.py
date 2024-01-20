# -- coding: utf-8 --
"""
Created on Fri Dec 22 08:00:21 2023

@author: sebas
"""

#importar librerias que se usaran
from numpy import array, loadtxt, mean, square, subtract
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# Definir módulos para las secuencias
def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # Hallar el fin de los patrones
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequence):
            break
    
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# Cargar datos desde el archivo de texto
raw_seq = loadtxt('mean_flow.txt')

n_steps_in, n_steps_out = 3, 6

X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)

# Adiestrar IA
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_steps_in))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=2000,batch_size =64, verbose=0)

# Más pruebas
x_input = array([1.620, 2.010, 2.790])  # Ajustar los valores según tus datos
x_input = x_input.reshape((1, n_steps_in))
y_hat = model.predict(x_input, verbose=0)
print("Valores predichos:", y_hat[0])

# Visualización de la serie original y valores predichos
plt.figure(figsize=(10, 6))

# Graficar la serie original
plt.plot(raw_seq, label='Serie Original', marker='o')

# Graficar los valores predichos
start_pred = len(raw_seq)
end_pred = start_pred + n_steps_out
time_pred = range(start_pred, end_pred)
plt.plot(time_pred, y_hat[0], label='Valores Predichos', linestyle='--', marker='x')

for i, txt in enumerate(y_hat[0]):
    plt.annotate(f'{txt:.2f}', (time_pred[i], y_hat[0][i]), textcoords="offset points", xytext=(0,10), ha='center')


# Añadir etiquetas y leyenda
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.title('Serie Original y Valores Predichos')
plt.legend()

# Mostrar la gráfica
plt.show()

#para hallar el error cuadrático medio

# Definir los valores observados y simulados
v_obs = array([25.230, 52.310, 146.150, 114.45, 39.84, 14.98])

# Tomar los valores simulados de y_hat
v_sim = y_hat[0]

# Calcular el error cuadrático
mse = mean(square(subtract(v_obs, v_sim)))

print(f'Error Cuadrático Medio (MSE): {mse}')
