# Blue XBee on Jetson
from digi.xbee.devices import XBeeDevice, RemoteXBeeDevice, XBee64BitAddress
import numpy as np

def my_data_received_callback(xbee_message):
    address = xbee_message.remote_device.get_64bit_addr()
    data = xbee_message.data.decode()
    data = np.fromstring(data[1:-1], sep=',')
    print(data)

# Setup Devices
local_device = XBeeDevice("/dev/ttyUSB0", 9600)
local_device.open()
remote_device = RemoteXBeeDevice(local_device, XBee64BitAddress.from_hex_string("0013A20041C1A0D8"))
local_device.add_data_received_callback(my_data_received_callback)

for i in range(100000):
    data = np.array([i])
    data = np.array2string(data, precision=2, separator=',',
                          suppress_small=True)
    local_device.send_data(remote_device, data)

local_device.close()
