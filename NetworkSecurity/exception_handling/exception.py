import sys

## We need to log this exception also
## Import logging

from NetworkSecurity.logging import logger

class NetworkSecurityException(Exception):
    def __init__(self, error_messsage, error_details:sys):
        self.error_message = error_messsage
        _,_,exc_tb = error_details.exc_info()

        self.lineno = exc_tb.tb_lineno
        self.file_name =  exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return 'Error occured in python script name [{0}] line number [{1}] error message [{2}]'.format(
        self.file_name, self.lineno, str(self.error_message))

if __name__ == '__main__':
    try:
        logger.logging.info('Enter the try block')
        a = 1/10
        print('This will not be printed', a)
    except Exception as e:
        logger.logging.info('Enter the exception block')
        raise NetworkSecurityException(e, sys)
    logger.logging.info('Exit the try - exception block')