import logging

class NativeLogger():
    def __init__(self):
        """
        Logger object initialization and setup
        """
        self.logger = logging.getLogger('drillDataLogging')
        logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    
    def getLogger(self):
        """Get logger object

        Returns:
            logging.Logger: Returns logger object
        """
        return self.logger
