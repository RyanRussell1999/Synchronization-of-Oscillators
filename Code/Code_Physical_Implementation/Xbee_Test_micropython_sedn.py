from digi.xbee.devices import XBeeDevice, RemoteXBeeDevice, XBee64BitAddress
import numpy as np

local_device = XBeeDevice("COM6", 9600)
local_device.open()

remote_device = RemoteXBeeDevice(local_device, XBee64BitAddress.from_hex_string("0013A20041C80821"))

data = np.array([10.1, 10, 20.3])

data = np.array2string(data, precision=2, separator=',',
                      suppress_small=True)

local_device.send_data(remote_device, data)

local_device.close()