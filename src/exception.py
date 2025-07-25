import sys
from src.logger import logging 

def error_message_detail(error_message, error_detail):
    _, _, exc_tb = error_detail.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        filename, exc_tb.tb_lineno, str(error_message)
    )
    
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
    

if __name__ == "__main__":

    try:
        a = 1/0
    except Exception as e:
        logging.info("Custom Exception test")
        raise CustomException(e, sys)

