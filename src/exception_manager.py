##################################################
#TFG - Gateway IOT con Edge ML sobre Linux Embebido con Yocto
#
#Author: Pablo Angresola Sanchez

##################################################

'''
Exception Module
'''

from enum import Enum

class ExceptionManagerType(Enum):
    GENERIC_ERROR = 0
    CONFIGURATION_ERROR = 1
    MQTT_ERROR = 2


class ExceptionManager(Exception):
    '''Main exception class for the MQTT-Serial comunication'''


    def __init__(self, message: str, exception_type: ExceptionManagerType = ExceptionManagerType.GENERIC_ERROR):
        """
        Initialize the exception
        
        Args:
            message: Error message
            error_code: Optional error code
        """
        self.message = message
        self.exception_type = exception_type
        super().__init__(self.message)


    def __str__(self):
        """
        String representation of the exception.

        Returns
        -------
        str
            Formatted error message
        """
        return f"[{self.exception_type.name}] {self.message}"


class ConfigurationError(ExceptionManager):
    """Exception raised for configuration-related errors."""
    
    def __init__(self, message: str):
        """
        Initialize the exception.

        Parameters
        ----------
        message : str
            Error message
        exception_type : ExceptionManagerType, optional
            Type of exception
        """
        super().__init__(message, ExceptionManagerType.CONFIGURATION_ERROR)


class MqttError(ExceptionManager):
    """Exception raised for MQTT-related errors."""
    
    def __init__(self, message: str):
        """
        Initialize the exception.

        Parameters
        ----------
        message : str
            Error message
        exception_type : ExceptionManagerType, optional
            Type of exception
        """
        super().__init__(message, ExceptionManagerType.MQTT_ERROR)