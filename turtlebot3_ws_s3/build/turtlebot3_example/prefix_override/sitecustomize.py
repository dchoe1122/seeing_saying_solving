import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/bchoe7/seeing_saying_solving/turtlebot3_ws_s3/install/turtlebot3_example'
