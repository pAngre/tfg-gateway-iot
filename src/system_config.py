##################################################
#TFG - Gateway IOT con Edge ML sobre Linux Embebido con Yocto
#
#Author: Pablo Angresola Sanchez

##################################################

'''
System Configuration Module

Configuration class, parsing the config file and providing an interface for the rest of classes
'''

import yaml
from pathlib import Path
from typing import Dict, List, Any

from exception_manager import ConfigurationError

class SystemConfig:
    """Configuration class for the Modbus to MQTT Gateway"""
    
    def __init__(self, config_path: str):
        """
        Initialize the system configuration.

        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file
        """
        self.__config_path = config_path
        self.__config_data = {}
        
        try:
            self.__load_config()
            self.__validate_config()
        except Exception as e:
            raise ConfigurationError(f"Failed to load or validate configuration: {e}")


    def __load_config(self):
        """
        Load and parse the YAML configuration file.

        Raises
        ------
        ConfigurationError
            If configuration file is not found or cannot be parsed
        """
        try:
            config_file = Path(self.__config_path)
            if not config_file.exists():
                raise ConfigurationError(f"Configuration file not found: {self.__config_path}")
            
            with open(config_file, 'r', encoding='utf-8') as f:
                self.__config_data = yaml.safe_load(f)
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse YAML configuration: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")


    def __validate_config(self):
        """
        Validate the configuration data.

        Raises
        ------
        ConfigurationError
            If required configuration sections or fields are missing
        """
        # Validate MQTT section
        if 'mqtt' not in self.__config_data:
            raise ConfigurationError("Missing 'mqtt' section in configuration")
        
        mqtt_config = self.__config_data['mqtt']
        required_mqtt_fields = ['broker', 'port', 'topics']
        for field in required_mqtt_fields:
            if field not in mqtt_config:
                raise ConfigurationError(f"Missing required MQTT field: {field}")

        # Validate Sensor section
        if 'sensor' not in self.__config_data:
            raise ConfigurationError("Missing 'sensor' section in configuration")

        sensor_config = self.__config_data['sensor']
        required_sensor_fields = ['fs', 'send_interval', 'freq', 'amp']
        for field in required_sensor_fields:
            if field not in sensor_config:
                raise ConfigurationError(f"Missing required Sensor field: {field}")
            
    # MQTT Configuration Properties
    @property
    def mqtt_broker(self) -> str:
        """
        Get MQTT broker address.

        Returns
        -------
        str
            MQTT broker address

        Raises
        ------
        ConfigurationError
            If MQTT broker configuration is missing
        """
        try:
            return self.__config_data['mqtt']['broker']
        except KeyError as e:
            raise ConfigurationError(f"Missing MQTT broker configuration: {e}")
            
    @property
    def mqtt_port(self) -> int:
        """
        Get MQTT broker port.

        Returns
        -------
        int
            MQTT broker port

        Raises
        ------
        ConfigurationError
            If MQTT port configuration is missing
        """
        try:
            return self.__config_data['mqtt']['port']
        except KeyError as e:
            raise ConfigurationError(f"Missing MQTT port configuration: {e}")

    @property
    def mqtt_topics(self) -> Dict[str, str]:
        """
        Get MQTT topics configuration.

        Returns
        -------
        Dict[str, str]
            MQTT topics configuration

        Raises
        ------
        ConfigurationError
            If MQTT topics configuration is missing
        """
        try:
            return self.__config_data['mqtt']['topics']
        except KeyError as e:
            raise ConfigurationError(f"Missing MQTT topics configuration: {e}")

    @property
    def mqtt_read_topic(self) -> str:
        """
        Get MQTT read topic.

        Returns
        -------
        str
            MQTT read topic

        Raises
        ------
        ConfigurationError
            If MQTT read topic configuration is missing
        """
        try:
            return self.__config_data['mqtt']['topics']['read']
        except KeyError as e:
            raise ConfigurationError(f"Missing MQTT read topic configuration: {e}")
    @property
    def mqtt_write_topic(self) -> str:
        """
        Get MQTT write topic.

        Returns
        -------
        str
            MQTT write topic

        Raises
        ------
        ConfigurationError
            If MQTT write topic configuration is missing
        """
        try:
            return self.__config_data['mqtt']['topics']['write']
        except KeyError as e:
            raise ConfigurationError(f"Missing MQTT write topic configuration: {e}")
            
    # Sensor Simulator Properties
    @property
    def sensor_fs(self) -> int:
        """
        Get Sensor sim sampling frequency.

        Returns
        -------
        int
            Sensor sampling frequency
        
        Raises
        ------
        ConfigurationError
            If Sampling frequency is missing
        """
        try:
            return self.__config_data['sensor']['fs']
        except KeyError as e:
            raise ConfigurationError(f"Missing Sensor Simulator sampling frequency: {e}")
    @property
    def sensor_send_interval(self) -> int:
        """
        Get Sensor sim send interval.

        Returns
        -------
        int
            Sensor send interval
        
        Raises
        ------
        ConfigurationError
            If Send interval is missing
        """
        try:
            return self.__config_data['sensor']['send_interval']
        except KeyError as e:
            raise ConfigurationError(f"Missing Sensor Simulator send interval: {e}")
    @property
    def sensor_freq(self) -> int:
        """
        Get Sensor sim frequency.

        Returns
        -------
        int
            Sensor frequency
        
        Raises
        ------
        ConfigurationError
            If Frequency is missing
        """
        try:
            return self.__config_data['sensor']['freq']
        except KeyError as e:
            raise ConfigurationError(f"Missing Sensor Simulator frequency: {e}")
    @property
    def sensor_amp(self) -> int:
        """
        Get Sensor sim amplitude.

        Returns
        -------
        int
            Sensor amplitude
        
        Raises
        ------
        ConfigurationError
            If Amplitude is missing
        """
        try:
            return self.__config_data['sensor']['amp']
        except KeyError as e:
            raise ConfigurationError(f"Missing Sensor Simulator amplitude: {e}")
    