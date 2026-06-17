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
import tflite_runtime.interpreter as tflite

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
        self.__interpreter = tflite.Interpreter(model_path="dataset/model.tflite")
        self.__interpreter.allocate_tensors()

        self.__input_details = self.__interpreter.get_input_details()
        self.__output_details = self.__interpreter.get_output_details()

        self.__mean = np.load("dataset/mean.npy")
        self.__std = np.load("dataset/std.npy")

        self.__pred_history = []
        self.__history_size = 3
        self.__pred_ant = 0


    def __mqtt_on_message(self, msg, topic):
        if topic == 'state':
            self.__sensor_sim.set_state(msg)

    
    def extract_features(self, signal):
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

        mapping = {
            "normal": 0,
            "fault_light": 1,
            "fault_severe": 2
        }

        self.__sensor_sim._SensorSimulator__freq = np.random.uniform(3, 7)
        self.__sensor_sim._SensorSimulator__amp = np.random.uniform(0.5, 2)

        while True:
            sample = self.__sensor_sim.get_reading()
            self.__buffer.append(sample)

            if len(self.__buffer) >= self.__system_config.sensor_send_interval:
                lat1 = time.time()
                features = self.extract_features(self.__buffer)
                features = np.array(features)
                features = (features - self.__mean) / self.__std
                features = features.astype(np.float32)
                features = np.expand_dims(features, axis=0)

                self.__interpreter.set_tensor(self.__input_details[0]['index'], features)
                self.__interpreter.invoke()

                output = self.__interpreter.get_tensor(self.__output_details[0]['index'])
                pred = int(np.argmax(output))
                lat = time.time() - lat1
                self.__pred_history.append(pred)
                if len(self.__pred_history) > self.__history_size:
                    self.__pred_history.pop(0)
                pred_final = max(set(self.__pred_history), key=self.__pred_history.count)

                payload = {
                    "data" : self.__buffer,
                    "length": len(self.__buffer),
                    "state": self.__sensor_sim.state,
                    "prediction": int(pred_final),
                    "latency": lat
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

