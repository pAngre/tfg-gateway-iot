import time
import json
import numpy as np
import paho.mqtt.client as mqtt
import tflite_runtime.interpreter as tflite

from exception_manager import ExceptionManager
from system_config import SystemConfig
from mqtt_interface import MqttInterface



class CloudMLTesting:
    def __init__(self, __conf_file):

        self.__system_config = SystemConfig(__conf_file)
        self.__mqtt_client = MqttInterface(self.__system_config, self.__on_message)

        # ===== MODELO =====
        self.__interpreter = tflite.Interpreter(model_path="dataset/model.tflite")
        self.__interpreter.allocate_tensors()

        self.__input_details = self.__interpreter.get_input_details()
        self.__output_details = self.__interpreter.get_output_details()

        self.__mean = np.load("dataset/mean.npy")
        self.__std = np.load("dataset/std.npy")

        # ===== MÉTRICAS =====
        self.__pred_history = []
        self.__history_size = 3
        self.__pred_ant = 0

        self.__mapping = {
            "normal": 0,
            "fault_light": 1,
            "fault_severe": 2
        }

    # ===== CALLBACK MQTT =====
    def __on_message(self, msg, topic):
        if topic == 'data':
            data = json.loads(msg)

            features = np.array(data["features"])
            features = (features - self.__mean) / self.__std
            features = features.astype(np.float32)
            features = np.expand_dims(features, axis=0)

            self.__interpreter.set_tensor(self.__input_details[0]['index'], features)
            self.__interpreter.invoke()

            output = self.__interpreter.get_tensor(self.__output_details[0]['index'])
            pred = int(np.argmax(output))

            self.__pred_history.append(pred)
            if len(self.__pred_history) > self.__history_size:
                self.__pred_history.pop(0)
            pred_final = max(set(self.__pred_history), key=self.__pred_history.count)

            payload = {
                "state": data["state"],
                "prediction": int(pred_final)
            }
            self.__mqtt_client.publish_reading(payload, topic='results')

    def run(self):
        self.__mqtt_client.connect()
        while(True):
            pass

if __name__ == "__main__":
    config = 'config.yaml'

    try:
        cloudPC = CloudMLTesting(config)
        cloudPC.run()
    except Exception as e:
        raise