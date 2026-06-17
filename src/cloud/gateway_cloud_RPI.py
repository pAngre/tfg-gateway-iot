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

    def __mqtt_on_message(self, msg, topic):
        if topic == 'state':
            self.__sensor_sim.set_state(msg)

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


        self.__sensor_sim._SensorSimulator__freq = np.random.uniform(3, 7)
        self.__sensor_sim._SensorSimulator__amp = np.random.uniform(0.5, 2)

        while True:
            sample = self.__sensor_sim.get_reading()
            self.__buffer.append(sample)

            if len(self.__buffer) >= self.__system_config.sensor_send_interval:
                features = self.__extract_features(self.__buffer)
                payload = {
                    "data": self.__buffer,
                    "features": features,
                    "state": self.__sensor_sim.state,
                    "timestamp": time.time()
                }
                self.__mqtt_client.publish_reading(payload)
                self.__buffer = []


            time.sleep(1 / self.__sensor_sim.fs)

if __name__ == "__main__":
    config = 'config.yaml'

    try:
        localML = LocalMLGateway(config)
        localML.run()
    except Exception as e:
        raise

