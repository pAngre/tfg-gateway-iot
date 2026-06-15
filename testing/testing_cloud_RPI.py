##################################################
#TFG - Gateway IOT con Edge ML sobre Linux Embebido con Yocto
#
#Author: Pablo Angresola Sanchez

##################################################

'''
Gateway IOT con modelo ML en local para la clasficacion de fallos.
'''
import time
import paho.mqtt.client as mqtt
import json
import joblib
import numpy as np

from exception_manager import ExceptionManager
from system_config import SystemConfig
from mqtt_interface import MqttInterface
from sensor_sim import SensorSimulator

class LocalMLGateway:
    def __init__(self, __conf_file):

        self.__system_config = SystemConfig(__conf_file)
        self.__mqtt_client = MqttInterface(self.__system_config, self.__mqtt_on_message)
        self.__sensor_sim = SensorSimulator(self.__system_config)
        self.__buffer = []
        self.__change_flag = False
        self.__change_time = None

        self.__freq = 0
        self.__amp = 0

        self.__normal = 0
        self.__fault_light = 0
        self.__fault_severe = 0

    def __mqtt_on_message(self, msg, topic):
        if topic == 'state':
            self.__sensor_sim.set_state(msg)
            self.__change_time = time.time()
            self.__change_flag = True
    
    def __change_state(self, new_state):
        self.__sensor_sim.set_state(new_state)
        self.__change_time = time.time()
        self.__change_flag = True

    def __extract_features(self, signal):
        signal = np.array(signal)

        return [
            np.mean(signal),
            np.std(signal),
            np.max(signal),
            np.min(signal),
            np.sum(signal**2),
            np.sqrt(np.mean(signal**2)),  # RMS
            np.percentile(signal, 75),
            np.percentile(signal, 25),
            np.mean(np.abs(signal)),
            np.max(signal) - np.min(signal),
            np.std(np.diff(signal))
        ]


    def run(self):
        self.__mqtt_client.connect()

        for _ in range(5):
            self.__sensor_sim._SensorSimulator__freq = np.random.uniform(3, 7)
            self.__sensor_sim._SensorSimulator__amp = np.random.uniform(0.5, 2)

            self.__freq = self.__sensor_sim._SensorSimulator__freq
            self.__amp = self.__sensor_sim._SensorSimulator__amp

            for _ in range(400):
                state = np.random.randint(0,3)
                if state == 0:
                    self.__change_state("normal")
                    self.__normal += 1
                elif state == 1:
                    self.__change_state("fault_light")
                    self.__fault_light += 1
                elif state == 2:
                    self.__change_state("fault_severe")
                    self.__fault_severe += 1

                for _ in range(1000):
                    sample = self.__sensor_sim.get_reading()
                    self.__buffer.append(sample)

                    if len(self.__buffer) >= self.__system_config.sensor_send_interval:
                        change_inf_time = time.time()
                        features = self.__extract_features(self.__buffer)
                        payload = {
                            "data": self.__buffer,
                            "features": features,
                            "state": self.__sensor_sim.state,
                            "change_flag": self.__change_flag,
                            "change_time": self.__change_time,
                            "timestamp": change_inf_time
                        }
                        self.__mqtt_client.publish_reading(payload, topic='data')
                        self.__buffer = []
                        self.__change_flag = False

                    time.sleep(1 / self.__sensor_sim.fs)

            print(f"Prubas con freq: {self.__freq:.2f} Hz y amp: {self.__amp:.2f}. Normal: {self.__normal}, Fault Light: {self.__fault_light}, Fault Severe: {self.__fault_severe}")
            self.__normal = 0
            self.__fault_light = 0
            self.__fault_severe = 0
        
        self.__mqtt_client.publish_reading({"test_complete": True}, topic='exit')

if __name__ == "__main__":
    config = '../config.yaml'

    try:
        localML = LocalMLGateway(config)
        localML.run()
    except Exception as e:
        raise
