##################################################
#TFG - Gateway IOT con Edge ML sobre Linux Embebido con Yocto
#
#Author: Pablo Angresola Sanchez

##################################################

'''

'''

import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, List

import paho.mqtt.client as mqtt

from exception_manager import MqttError, ConfigurationError
from system_config import SystemConfig

class MqttInterface:
    """MQTT gateway class."""

    # Timeout for MQTT connection in seconds
    CONNECTION_TIMEOUT_SEC = 10

    # MQTT keepalive in seconds
    KEEPALIVE_SEC = 60

    def __init__(self, config: SystemConfig, on_message=None):
        """
        Initialize the MQTT interface.

        Parameters
        ----------
        config : SystemConfig
            System configuration instance
        on_message : callable, optional
            Callback for incoming messages

        Raises
        ------
        MqttError
            If MQTT client setup fails
        """
        self.__config = config

        
        self.__on_message = on_message

        self.__broker = config.mqtt_broker
        self.__port = config.mqtt_port
        self.__connected = False

        self.__subscription_topic = config.mqtt_read_topic
        self.__publication_topic = config.mqtt_write_topic
       
        try:
            self.__setup_mqtt_client()
        except MqttError:
            raise


    def __setup_mqtt_client(self):
        """Setup the MQTT client."""
        try:
            self.__client = mqtt.Client()
            self.__client.on_connect = self.__on_connect
            self.__client.on_disconnect = self.__on_disconnect
            self.__client.on_message = self.__on_message_received
            
        except ConfigurationError:
            raise
    
    def connect(self):
        """Connect to the MQTT broker."""
        try: 
            self.__client.connect(self.__broker, self.__port, keepalive=MqttInterface.KEEPALIVE_SEC)
            self.__client.loop_start()
        except Exception as e:
            raise MqttError(f"Failed to connect to MQTT broker: {e}")

        # Wait for connection
        start_time = time.time()
        while not self.__connected and (time.time() - start_time) < MqttInterface.CONNECTION_TIMEOUT_SEC:
            time.sleep(0.1)
        
        if not self.__connected:
            raise MqttError("Failed to connect to MQTT broker within timeout")
        
    def disconnect(self):
        """Disconnect from the MQTT broker."""
        if self.__client:
            self.__client.disconnect()

    def __on_connect(self, client, userdata, flags, rc):
        """
        Callback for when the client connects to the broker.

        Parameters
        ----------
        client : mqtt.Client
            MQTT client instance
        userdata : Any
            User data
        flags : dict
            Connection flags
        rc : int
            Result code

        Raises
        ------
        MqttError
            If susbscription to write topic fails
        """
        if rc == 0:
            self.__connected = True
            
            try:
                # Subscribe to write topic
                self.__client.subscribe(self.__subscription_topic)
            except Exception as e:
                raise MqttError(f"Failed to subscribe to write topic: {e}")
        else:
            pass

    def __on_disconnect(self, client, userdata, rc):
        """
        Callback for when the client disconnects from the broker.

        Parameters
        ----------
        client : mqtt.Client
            MQTT client instance
        userdata : Any
            User data
        rc : int
            Result code
        """
        self.__connected = False

    def __on_message_received(self, client, userdata, msg):
        """
        Callback for when a message is received.

        Parameters
        ----------
        client : mqtt.Client
            MQTT client instance
        userdata : Any
            User data
        msg : mqtt.MQTTMessage
            MQTT message
        """
        try:
            # Parse the message
            payload = msg.payload.decode()
            topic = msg.topic.rstrip('/').split('/')[-1]
            self.__on_message(payload, topic)
            
        except json.JSONDecodeError:
            pass
        except Exception:
            pass
             

    def publish_reading(self, data: {}, topic = None):
        """
        Publish reading over MQTT.

        Parameters
        ----------
        data : Dict[str, str]
            data to send
        """
        if topic != None:
            topic = f"{self.__publication_topic}/{topic}"
        else:
            topic = self.__publication_topic
        try:
            self.__client.publish(topic, json.dumps(data), retain=True)
            
        except Exception as ex:
            raise MqttError(f"Failed to publish MQTT message: {ex}")


    
