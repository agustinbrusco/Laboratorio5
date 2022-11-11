# -*- coding: utf-8 -*-
"""
Created on Wed May  4 16:16:04 2022

@author: Publico
"""

import matplotlib.pyplot as plt
import numpy as np
import nidaqmx
import math
import time


#para saber el ID de la placa conectada (DevX)
system = nidaqmx.system.System.local()
for device in system.devices:
    print(device)

#para setear (y preguntar) el modo y rango de un canal analógico
with nidaqmx.Task() as task:
    ai_channel = task.ai_channels.add_ai_voltage_chan("Dev2/ai1",max_val=10,min_val=-10)
    print(ai_channel.ai_term_cfg)    
    print(ai_channel.ai_max)
    print(ai_channel.ai_min)	
	

## Medicion por tiempo/samples de una sola vez
def medicion_una_vez(duracion, fs):
    cant_puntos = int(duracion*fs)
    with nidaqmx.Task() as task:
        modo= nidaqmx.constants.TerminalConfiguration.BAL_DIFF
        task.ai_channels.add_ai_voltage_chan("Dev2/ai1", terminal_config = modo)
               
        task.timing.cfg_samp_clk_timing(fs,samps_per_chan = cant_puntos,
                                        sample_mode = nidaqmx.constants.AcquisitionType.FINITE)
        
        datos = task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)           
    datos = np.asarray(datos)    
    return datos

duracion = 1 #segundos
fs = 1000 #Frecuencia de muestreo
y = medicion_una_vez(duracion, fs)
t = np.linspace(0,.999,1000)
plt.plot(t,y)
plt.grid()
plt.show()


for i in range(len(y)):
    fsalida = open('MedicionesVoltajes.txt','a')
    fsalida.write(str(t[i]) + ' ; ')
    fsalida.write(str(y[i])+ '\n')
    fsalida.close()