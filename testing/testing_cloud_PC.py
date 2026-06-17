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

        self.__ant_iter = 1

        self.__total = 0
        self.__fallo = 0
        self.__fallo_basic = 0
        self.__avg_inf_latency = []
        self.__avg_inf_red_latency = []
        self.__avg_latency = []
        self.__total_changes = 0

        self.__total_iter = 0
        self.__fallo_iter = 0
        self.__fallo_basic_iter = 0
        self.__avg_inf_latency_iter = []
        self.__avg_inf_red_latency_iter = []
        self.__avg_latency_iter = []
        self.__total_changes_iter = 0

        self.__iteration_results = []

        self.__normal = 0
        self.__fault_light = 0
        self.__fault_severe = 0

        self.__change_flag = False


        self.__mapping = {
            "normal": 0,
            "fault_light": 1,
            "fault_severe": 2
        }

    def __reset_metrics(self):
        self.__total_iter = 0
        self.__fallo_iter = 0
        self.__fallo_basic_iter = 0
        self.__avg_inf_latency_iter = []
        self.__avg_latency_iter = []
        self.__total_changes_iter = 0

        self.__normal = 0
        self.__fault_light = 0
        self.__fault_severe = 0

        self.__pred_history = []

    # ===== CALLBACK MQTT =====
    def __on_message(self, msg, topic):
        if topic == 'data':
            data = json.loads(msg)

            if data["iteration"] != self.__ant_iter:
                iter_results = {
                    "iteration": self.__ant_iter,
                    "fallos": self.__fallo_iter,
                    "accuracy": round(((self.__total_iter - self.__fallo_iter) / self.__total_iter) * 100, 4),
                    "fallos_basicos": self.__fallo_basic_iter,
                    "accuracy_basica": round(((self.__total_iter - self.__fallo_basic_iter) / self.__total_iter) * 100, 4),
                    "avg_inf_latency": round(np.mean(self.__avg_inf_latency_iter), 4),
                    "avg_latency": round(np.mean(self.__avg_latency_iter), 4),
                    "total_changes": self.__total_changes_iter,
                    "total_samples": self.__total_iter
                }
                self.__iteration_results.append(iter_results)
                self.__ant_iter = data["iteration"]
                self.__reset_metrics()

            if data["change_flag"]:
                self.__change_flag = True
                self.__total_changes += 1
                self.__total_changes_iter += 1

                if data["state"] == "normal":
                    self.__normal += 1
                elif data["state"] == "fault_light":
                    self.__fault_light += 1
                elif data["state"] == "fault_severe":
                    self.__fault_severe += 1

            change_inf_time = time.time()

            features = np.array(data["features"])
            features = (features - self.__mean) / self.__std
            features = features.astype(np.float32)
            features = np.expand_dims(features, axis=0)

            self.__interpreter.set_tensor(self.__input_details[0]['index'], features)
            self.__interpreter.invoke()

            output = self.__interpreter.get_tensor(self.__output_details[0]['index'])
            pred = int(np.argmax(output))

            inf_latency = time.time() - change_inf_time
            red_inf_latency = time.time() - data["timestamp"]
            self.__avg_inf_latency.append(inf_latency)
            self.__avg_inf_latency_iter.append(inf_latency)
            self.__avg_inf_red_latency.append(red_inf_latency)
            self.__avg_inf_red_latency_iter.append(red_inf_latency)
            self.__total += 1
            self.__total_iter += 1

            self.__pred_history.append(pred)
            if len(self.__pred_history) > self.__history_size:
                self.__pred_history.pop(0)
            pred_final = max(set(self.__pred_history), key=self.__pred_history.count)

            if self.__change_flag and pred_final == self.__mapping[data["state"]]:
                self.__change_flag = False
                change_latency = time.time() - data["change_time"]
                self.__avg_latency.append(change_latency)
                self.__avg_latency_iter.append(change_latency)

            if pred != self.__mapping[data["state"]]:
                self.__fallo_basic += 1
                self.__fallo_basic_iter += 1

            if pred_final != self.__mapping[data["state"]] and not self.__change_flag:
                self.__fallo += 1
                self.__fallo_iter += 1


        elif topic == 'exit':
            iter_results = {
                    "iteration": self.__ant_iter,
                    "fallos": self.__fallo_iter,
                    "accuracy": round(((self.__total_iter - self.__fallo_iter) / self.__total_iter) * 100, 4),
                    "fallos_basicos": self.__fallo_basic_iter,
                    "accuracy_basica": round(((self.__total_iter - self.__fallo_basic_iter) / self.__total_iter) * 100, 4),
                    "avg_inf_latency": round(np.mean(self.__avg_inf_latency_iter), 4),
                    "avg_inf_red_latency": round(np.mean(self.__avg_inf_red_latency_iter), 4),
                    "avg_latency": round(np.mean(self.__avg_latency_iter), 4),
                    "total_changes": self.__total_changes_iter,
                    "total_samples": self.__total_iter
            }
            self.__iteration_results.append(iter_results)

            acc_values = [r["accuracy"] for r in self.__iteration_results]
            acc_basic_values = [r["accuracy_basica"] for r in self.__iteration_results]
            lat_inf_values = [r["avg_inf_latency"] for r in self.__iteration_results]
            lat_det_values = [r["avg_latency"] for r in self.__iteration_results]
            lat_inf_red_values = [r["avg_inf_red_latency"] for r in self.__iteration_results]

            payload = {
                "accuracy": ((self.__total - self.__fallo) / self.__total) * 100,
                "fallos": self.__fallo,
                "accuracy_basica": ((self.__total - self.__fallo_basic) / self.__total) * 100,
                "fallos_basicos": self.__fallo_basic,
                "avg_inf_red_latency": np.mean(self.__avg_inf_red_latency),
                "avg_inf_latency": np.mean(self.__avg_inf_latency),
                "avg_latency": np.mean(self.__avg_latency),
                "total_samples": self.__total,
                "total_changes": self.__total_changes,
                "std_accuracy": round(float(np.std(acc_values)), 4),
                "std_accuracy_basica": round(float(np.std(acc_basic_values)), 4),
                "std_avg_inf_latency": round(float(np.std(lat_inf_values)), 4),
                "std_avg_inf_red_latency": round(float(np.std(lat_inf_red_values)), 4),
                "std_avg_latency": round(float(np.std(lat_det_values)), 4),
                "min_accuracy": round(float(np.min(acc_values)), 4),
                "max_accuracy": round(float(np.max(acc_values)), 4),
                "iteration_results": self.__iteration_results
            }
            print(json.dumps(payload, indent=4))
            with open('results/results_cloud.json', 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=4)

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