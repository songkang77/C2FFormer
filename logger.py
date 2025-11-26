import os
import sys
from datetime import datetime

def setup_logging(log_dir='logs'):
    """
    
    
    Args:
        log_dir (str): 
    
    Returns:
        str: 
    """
    # 
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_log_{timestamp}.txt')
    
    # 
    log_file_handler = open(log_file, 'w')
    
    # 
    original_stdout = sys.stdout
    
    # 
    class Logger:
        def __init__(self, file, stdout):
            self.file = file
            self.stdout = stdout
        
        def write(self, message):
            self.file.write(message)
            self.stdout.write(message)
            # 
            self.file.flush()
        
        def flush(self):
            self.file.flush()
            self.stdout.flush()
    
    # 
    sys.stdout = Logger(log_file_handler, original_stdout)
    
    print(f"logger file: {log_file}")
    return log_file
