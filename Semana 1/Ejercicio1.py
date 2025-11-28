import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

model = keras.Sequential([
    layers.Dense(1, input_shape=(4,), activation='sigmoid', use_bias=True)
])

pesos = np.array([[4.0],   # Descuento
                  [7.0],   # Necesidad
                  [5.0],   # Presupuesto
                  [2.0]])  # Impulsividad

sesgo = np.array([-6.0])

model.layers[0].set_weights([pesos, sesgo])

entrada = np.array([[0.1, 0.1, 0.9, 0.1]])  

salida = model.predict(entrada)

print(f"Valor de activación: {salida[0][0]:.4f}") #z=(x1​⋅w1​)+(x2​⋅w2​)+(x3​⋅w3​)+(x4​⋅w4​)+b
if salida[0][0] >= 0.5:
    print("✅ Comprar: Vale la pena aprovechar la oferta.")
else:
    print("❌ No comprar: Mejor esperar o buscar otra opción.")
