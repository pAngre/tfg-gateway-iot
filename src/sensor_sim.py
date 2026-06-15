##################################################
#TFG - Gateway IOT con Edge ML sobre Linux Embebido con Yocto
#
#Author: Pablo Angresola Sanchez

##################################################

'''
Simulador de Sensor para la clasificacion de fallos.
- Clase 0: Normal
- Clase 1: Fallo leve (ruido agregado)
- Clase 2: Fallo severo (ruido agregado + picos)
'''

import numpy as np

from exception_manager import MqttError, ConfigurationError
from system_config import SystemConfig

class SensorSimulator:
    def __init__(self, config: SystemConfig):
        self.__config = config
        self.__freq = self.__config.sensor_freq
        self.__amp= self.__config.sensor_amp
        self.fs = self.__config.sensor_fs
        self.__dt = 1 / self.fs
        self.__t = 0
        self.state = "normal"

    def _base_signal(self) -> np.ndarray:
        return self.__amp * np.sin(2 * np.pi * self.__freq * self.__t)

    def _generate_normal(self) -> np.ndarray:
        return self._base_signal()

    def _generate_fault_light(self) -> np.ndarray:
        signal = self._base_signal()
        noise = np.random.uniform(0.1, 0.3) * np.random.randn()
        amp_mod = 1 + 0.1 * np.sin(2 * np.pi * 0.5 * self.__t)

        return amp_mod *signal + noise
    
    def _generate_fault_severe(self) -> np.ndarray:
        signal = self._base_signal()
        noise = np.random.uniform(0.4, 0.8) * np.random.randn()

        harmonic = 0.5 * np.sin(2 * np.pi * (self.__freq * 2) * self.__t)

        spike = 0
        if np.random.rand() < 0.05:
            spike = np.random.uniform(2, 5)

        return signal + harmonic + noise + spike

    def set_state(self, state: str):
        if state not in ["normal", "fault_light", "fault_severe"]:
            raise ValueError("Not a valid state. Use 'normal', 'fault_light' or 'fault_severe'.")
        self.state = state

    def get_reading(self):
        if self.state == "normal":
            value = self._generate_normal()
        elif self.state == "fault_light":
            value = self._generate_fault_light()
        elif self.state == "fault_severe":
            value = self._generate_fault_severe()

        self.__t += self.__dt
        return value

        
