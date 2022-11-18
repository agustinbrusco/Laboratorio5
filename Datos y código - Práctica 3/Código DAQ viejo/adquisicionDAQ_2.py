# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 16:37:12 2022

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
    
## Medicion continua
def medicion_continua(duracion, fs):
    cant_puntos = int(duracion*fs)
    with nidaqmx.Task() as task:
        
        modo= nidaqmx.constants.TerminalConfiguration.BAL_DIFF
        ai_channel1 = task.ai_channels.add_ai_voltage_chan("Dev4/ai1", terminal_config = modo,max_val=10)
        modo= nidaqmx.constants.TerminalConfiguration.BAL_DIFF
        ai_channel2 = task.ai_channels.add_ai_voltage_chan("Dev4/ai3", terminal_config = modo,max_val=10)
        print(ai_channel1.ai_term_cfg)    
        print(ai_channel2.ai_term_cfg)    
        
        task.timing.cfg_samp_clk_timing(fs, sample_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS)
        task.start()
        t0 = time.time()
        total = 0
        V1 =[]
        V2 =[]
        tiempo_ =[]
        while total<cant_puntos:
            time.sleep(0.2)
            datos = task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)                       
            t1 = time.time()
            datos = np.asarray(datos)    
            volt1 = datos[0,:]
            volt2 = datos[1,:]            
            V1.extend(volt1)
            V2.extend(volt2)
            total = total + len(volt1)
            t_ = t1 - t0
            tiempo_.append(t_)
            print("%2.3fs %d %d %2.3f" % (t1-t0, len(volt1), total, total/(t1-t0)))            
        return V1,V2, tiempo_

fs = 1000 #Frecuencia de muestreo
duracion = 180 #segundos
V1,V2, t=medicion_continua(duracion, fs)

tiempo = np.linspace(t[0],t[-1],len(V1) )

plt.figure()
plt.title('Fotodiodo')
plt.plot(tiempo, V2)
plt.grid()
plt.show()

plt.figure()
plt.title('Termocupla')
plt.plot(tiempo, V1)
plt.grid()
plt.show()

plt.figure()
plt.title('Ambos')
plt.plot(tiempo, V1)
plt.plot(tiempo, V2)
plt.grid()
plt.show()

data = np.array([V1, V2, tiempo]).T

np.savetxt('data1.csv', data, delimiter=';')

#np.savetxt("termocupla1.csv", V1, delimiter=";")
#np.savetxt("fotodiodo1.csv", V2, delimiter=";")
#np.savetxt("tiempo1.csv", tiempo, delimiter=";")

#for i in range(len(V1)):
#    fsalida = open(f'MedicionesVoltajes_Termocupla_{duracion}_segs1.txt','a')
#    fsalida.write(str(tiempo[i]) + ' ; ')
#    fsalida.write(str(V1[i])+ '\n')
#    fsalida.close()
    
#for i in range(len(V2)):
#    fsalida = open(f'MedicionesVoltajes_Fotodiodo_{duracion}_segs1.txt','a')
#    fsalida.write(str(tiempo[i]) + ' ; ')
#    fsalida.write(str(V2[i])+ '\n')
#    fsalida.close()

