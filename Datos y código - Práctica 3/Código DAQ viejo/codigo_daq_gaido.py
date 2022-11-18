"""
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import nidaqmx
import matplotlib
import matplotlib.pyplot as plt

path = str(Path(__file__).parent.absolute().absolute())

# --- #


def medicion_una_vez(duracion, fs, dev_ch='Dev1/ai3'):

    cant_puntos = int(duracion * fs)
    with nidaqmx.Task() as task:
        modo = nidaqmx.constants.TerminalConfiguration.DIFF
        task.ai_channels.add_ai_voltage_chan(dev_ch, terminal_config=modo)

        task.timing.cfg_samp_clk_timing(fs,
                                        samps_per_chan=cant_puntos,
                                        sample_mode=nidaqmx.constants.AcquisitionType.FINITE)

        data = task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)
    data = np.asarray(data)
    return data


def medicion_continua(duracion, fs):
    cant_puntos = int(duracion * fs)
    with nidaqmx.Task() as task:
        modo = nidaqmx.constants.TerminalConfiguration.DIFF
        task.ai_channels.add_ai_voltage_chan(dev_ch, terminal_config=modo)
        task.timing.cfg_samp_clk_timing(fs, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
        task.start()
        t0 = time.time()
        total = 0
        while total < cant_puntos:
            time.sleep(0.1)
            datos = task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)
            total = total + len(datos)
            t1 = time.time()
            print("%2.3fs %d %d %2.3f" % (t1 - t0, len(datos), total, total / (t1 - t0)))



def measure_save(set_name,
                 list_params,
                 keys_params,
                 n_sets,
                 set_duration,
                 frequency,
                 input_range,
                 dev_ch,
                 destiny_folder):

    """
    :param set_name:
    :param list_params:
    :param keys_params:
    :param n_sets:
    :param set_duration:
    :param frequency:
    :param input_range:
    :param dev_ch:
    :param destiny_folder:
    :return:

    """

    with nidaqmx.Task() as task:
        ai_channel = task.ai_channels.add_ai_voltage_chan(dev_ch,
                                                          max_val=input_range,
                                                          min_val=-input_range)
        print(ai_channel.ai_term_cfg)
    #    print(ai_channel.ai_max)

    print(f'\n')
    print('#=============#')
    print(set_name)
    print('#=============#')

    if not os.path.exists(destiny_folder):
        os.makedirs(destiny_folder)
        print(f'carpeta destino: "{set_name}"\n-----------')

    else:
        print('Carpeta ya existe')
        for file in os.listdir(destiny_folder):
            os.remove(destiny_folder + f'/{file}')

    list_params += [n_sets, set_duration, frequency, input_range]
    keys_params += ['Cantidad de mediciones',
                    'Duración de cada medición [s]',
                    'Frecuencia de muestreo [Hz]',
                    'Rango operativo [V]']

    dict_params = dict(zip(keys_params, list_params))

    string_table = f'Lista Parámetros: \n'
    for k, v in dict_params.items():
        new_line = f'{k} = {v} \n'
        string_table += new_line

    path_params = destiny_folder + f'/parametros_medicion.txt'

    with open(path_params, 'w') as text_file:
        print(string_table)
        text_file.write(string_table)

    input('Parámetros ok? Esperando confirmación')

    # input(f'Nombre de medición: {medicion} ok?')
    # input(f'Parámetros ok?')

    for j in tqdm(range(0, n_sets)):

        tension = medicion_una_vez(set_duration, frequency)
        tiempo = np.arange(0, set_duration, 1 / frequency)
        if len(tiempo) != len(tension):
            tiempo = np.arange(0, set_duration - 1 / frequency, 1 / frequency)

        # plt.figure(1)
        # plt.plot(tiempo, tension)
        # plt.xlabel('Tiempo [s]')
        # plt.ylabel('Voltaje [V]')
        # # plt.ylim()
        # plt.show()

        # plt.figure(2)
        # plt.hist(tension, bins=25)
        # plt.show()

        file = destiny_folder + f'/file_{j + 1}.txt'
        data = np.array([tiempo, tension]).T
        np.savetxt(file, data, delimiter=',', header="Tiempo,Tensión")

    print('Mediciones registradas')
    return None


"""
        SENSOR DAQ

Resolución

    - 16 Bits

Rangos operativos permitidos:

    - (-10, 10) V
    - (-5, 5) V
    - (-1, 1) V
    - (-200, 200) mV

Frecuencia de Muestreo:
    
    - Hasta 250_000 Hz
    

Resistencia en pines de antena
        
"""

medicion = 'fotones_r2_daq'
sub_folder = path_mediciones + f'/{medicion}'

dev_ch = 'Dev1/ai3'
input_range = 1

# escala_t = 2.5e-2
# escala_v = 0.2
# zero_tension = 3

res_daq = 47_000
tension_fotomult = -1050

list_params = [medicion, res_daq, tension_fotomult] #, escala_t, escala_v, zero_tension]
keys_params = ['NOMBRE',
               'Resistencia DAQ [Ohm]: ',
               'Tensión FM [V]: ']
               # 'Escala temporal',
               # 'Escala tensión',
               # 'Cero de tensión']


N = 1000  # cantidad de conjuntos
fs = 250_000  # frecuencia de muestreo
duracion = 0.25  # segundos
input_range = 1  # en volts

ut.measure_save(set_name=medicion,
                list_params=list_params,
                keys_params=keys_params,
                n_sets=N,
                set_duration=duracion,
                frequency=fs,
                input_range=input_range,
                dev_ch=dev_ch,
                destiny_folder=sub_folder)