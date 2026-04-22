import pyvisa
import support_functions as sf
import time
import os
import csv
import math

addr = 'TCPIP0::192.168.1.10::inst0::INSTR'
VNA_model = 'E5071C'

low_freq = 300e6
upper_freq = 6e9

print('\nAttemping to connect to: ' + sf.get_yellow(addr))

rm = pyvisa.ResourceManager()
VNA = rm.open_resource(addr)

VNA.write_termination = "\n"
VNA.read_termination = None
VNA.timeout = 20000

resp = sf.query_text(VNA, "*IDN?")


if VNA_model in resp:
    sf.print_green('Successfully connected to Keysight VNA simulator!\n')
else:
    sf.print_red('Unable to connect / model mismatch.\n')

VNA.write("*RST")
