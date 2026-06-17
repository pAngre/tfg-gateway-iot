import numpy as np
from sensor_sim import SensorSimulator
from system_config import SystemConfig

def extract_features(signal):
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


def generate_dataset(config_path, samples_per_class=1000):
    config = SystemConfig(config_path)
    sensor = SensorSimulator(config)

    X = []
    y = []

    states = {
        "normal": 0,
        "fault_light": 1,
        "fault_severe": 2
    }

    for state, label in states.items():
        print(f"Generando clase: {state}")

        for _ in range(int(samples_per_class)):
            sensor.set_state(state)
            
            sensor._SensorSimulator__freq = np.random.uniform(3, 7)
            sensor._SensorSimulator__amp = np.random.uniform(0.5, 2)

            for _ in range(200):
                sensor.get_reading()

            window = []

            for _ in range(config.sensor_send_interval):
                window.append(sensor.get_reading())

            features = extract_features(window)

            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)


if __name__ == "__main__":
    X, y = generate_dataset("config.yaml", samples_per_class=1000)

    np.save("dataset/X.npy", X)
    np.save("dataset/y.npy", y)

    print("Dataset generado:", X.shape, y.shape)