import sys
import types


# Format error trace
def format_error(error: Exception, error_detail: types.ModuleType) -> str:
    exc_info = error_detail.exc_info()
    exc_tb = exc_info[2] if exc_info else None
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_num = exc_tb.tb_lineno
        return (
            f"Error occurred in script: [{file_name}] "
            f"at line number: [{line_num}] "
            f"with error message: [{error!s}]"
        )
    return f"Error: [{error!s}]"

# Custom exception class
class CustomException(Exception):
    def __init__(self, error_msg: Exception, error_detail: types.ModuleType = sys):
        super().__init__(str(error_msg))
        self.error_message = format_error(error_msg, error_detail=error_detail)

    def __str__(self) -> str:
        return self.error_message
