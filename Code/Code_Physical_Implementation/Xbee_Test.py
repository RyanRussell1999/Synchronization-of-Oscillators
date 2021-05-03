from digi.xbee.devices import XBeeDevice, RemoteXBeeDevice, XBee64BitAddress

local_device = XBeeDevice("COM6", 9600)
local_device.open()

remote_device = RemoteXBeeDevice(local_device, XBee64BitAddress.from_hex_string("0013A20041C80821"))

local_device.send_data(remote_device, "Hello World")

local_device.close()