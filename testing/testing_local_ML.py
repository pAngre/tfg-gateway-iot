##################################################
#TFG - Gateway IOT con Edge ML sobre Linux Embebido con Yocto
#
#Author: Pablo Angresola Sanchez

##################################################

'''
Gateway IOT con modelo ML en local para la clasficacion de fallos.
'''
import time
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

        self.__total = 0
        self.__fallo = 0
        self.__fallo_basic = 0
        self.__acc = 1
        self.__acc_basic = 1
        self.__avg_inf_latency = []

        self.__last_state = None
        self.__change_time = None
        self.__waiting = False
        self.__avg_latency = []
        self.__total_changes = 0


        self.__normal = 0
        self.__fault_light = 0
        self.__fault_severe = 0

        self.__freq = 0
        self.__amp = 0

    
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

        mapping = {
            "normal": 0,
            "fault_light": 1,
            "fault_severe": 2
        }

        for _ in range(5):
            self.__freq = np.random.uniform(3, 7)
            self.__amp = np.random.uniform(0.5, 2)

            self.__sensor_sim._SensorSimulator__freq = self.__freq
            self.__sensor_sim._SensorSimulator__amp = self.__amp

            for _ in range(400):
                state = np.random.randint(0,3)
                if state == 0:
                    self.__sensor_sim.set_state("normal")
                    self.__normal += 1
                elif state == 1:
                    self.__sensor_sim.set_state("fault_light")
                    self.__fault_light += 1
                elif state == 2:
                    self.__sensor_sim.set_state("fault_severe")
                    self.__fault_severe += 1

                current_state = state

                if current_state != self.__last_state:
                    self.__change_time = time.time()
                    self.__waiting = True
                    self.__last_state = current_state
                    self.__total_changes += 1

                for _ in range(1000):
                    sample = self.__sensor_sim.get_reading()
                    self.__buffer.append(sample)

                    if len(self.__buffer) >= self.__system_config.sensor_send_interval:
                        change_inf_time = time.time()
                        features = self.extract_features(self.__buffer)
                        features = np.array(features)
                        features = (features - self.__mean) / self.__std
                        features = features.astype(np.float32)
                        features = np.expand_dims(features, axis=0)

                        self.__interpreter.set_tensor(self.__input_details[0]['index'], features)
                        self.__interpreter.invoke()

                        output = self.__interpreter.get_tensor(self.__output_details[0]['index'])
                        pred = int(np.argmax(output))

                        inf_latency = time.time() - change_inf_time

                        self.__avg_inf_latency.append(inf_latency)
                        self.__total += 1

                        self.__pred_history.append(pred)
                        if len(self.__pred_history) > self.__history_size:
                            self.__pred_history.pop(0)
                        pred_final = max(set(self.__pred_history), key=self.__pred_history.count)
                        
                        if self.__waiting and pred_final == current_state:
                            latency = time.time() - self.__change_time
                            self.__avg_latency.append(latency)
                            self.__waiting = False

                        if pred_final != mapping[self.__sensor_sim.state] and not self.__waiting:
                            self.__fallo += 1

                        if pred != mapping[self.__sensor_sim.state]:
                            self.__fallo_basic += 1
                            
                        self.__buffer = []

                    time.sleep(1 / self.__sensor_sim.fs)

            print(f"Prubas con freq: {self.__freq:.2f} Hz y amp: {self.__amp:.2f}. Normal: {self.__normal}, Fault Light: {self.__fault_light}, Fault Severe: {self.__fault_severe}")
            self.__normal = 0
            self.__fault_light = 0
            self.__fault_severe = 0
        
        payload = {
                    "accuracy": ((self.__total - self.__fallo) / self.__total) * 100,
                    "fallos": self.__fallo,
                    "fallos_basicos": self.__fallo_basic,
                    "accuracy_basica": ((self.__total - self.__fallo_basic) / self.__total) * 100,
                    "avg_inf_latency": np.mean(self.__avg_inf_latency) if self.__avg_inf_latency else 0,
                    "avg_latency": np.mean(self.__avg_latency) if self.__avg_latency else 0,
                    "total_changes": self.__total_changes,
                    "total_samples": self.__total
                }
        print(json.dumps(payload, indent=4))
        with open('results.json', 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    config = 'config.yaml'

    try:
        localML = LocalMLGateway(config)
        localML.run()
    except Exception as e:
        raise
