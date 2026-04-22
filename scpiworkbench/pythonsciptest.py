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

s_params_list = ['11']

reset_to_instrument_preset = False

manual_step_size = False
step_freq = 1e6

pc_dir = r"C:\Users\ericd\Desktop\SCPI FOLDER\\"

print('\nAttemping to connect to: ' + sf.get_yellow(addr))

rm = pyvisa.ResourceManager()
VNA = rm.open_resource(addr)

VNA.write_termination = "\n"
VNA.read_termination = None
VNA.timeout = 20000

resp = sf.query_text(VNA, "*IDN?")
VNA.write(f"CALC1:PAR:COUN {len(s_params_list)}")

if VNA_model in resp:
    sf.print_green('Successfully connected to Keysight VNA simulator!\n')
else:
    sf.print_red('Unable to connect / model mismatch.\n')

if reset_to_instrument_preset:
    sf.toggle_preset(VNA)
    VNA.write("*CLS")

VNA.write(f"CALC1:PAR:COUN {len(s_params_list)}")
VNA.write(f"SENS1:FREQ:STAR {low_freq}")
VNA.write(f"SENS1:FREQ:STOP {upper_freq}")

if manual_step_size:
    npts = int(round((upper_freq - low_freq) / step_freq)) + 1
    if npts < 2:
        npts = 2
    VNA.write(f"SENS1:SWE:POIN {npts}")

VNA.write("*CLS")
VNA.write("INIT1:CONT OFF")

for trace_idx, s_param in enumerate(s_params_list, start=1):
    meas = f"S{s_param}"

    VNA.write(f"CALC1:PAR{trace_idx}:DEF {meas}")
    VNA.write(f"DISP:WIND1:TRAC{trace_idx}:FEED '{meas}'")
    VNA.write("DISP:WIND1:ACT")
    VNA.write(f"CALC1:PAR{trace_idx}:SEL")

    VNA.write("INIT1:IMM")
    time.sleep(0.2)

    freqs = sf.query_csv_numbers(VNA, "SENS1:FREQ:DATA?")
    sdata = sf.query_csv_numbers(VNA, "CALC1:DATA:SDAT?")  # Re0,Im0,Re1,Im1,...

    if len(sdata) != 2 * len(freqs):
        raise RuntimeError(f"Mismatch: freqs={len(freqs)} sdata={len(sdata)}")

    local_csv = os.path.join(pc_dir, f"{meas}_data.csv")
    os.makedirs(os.path.dirname(local_csv), exist_ok=True)

    with open(local_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Frequency_Hz", "Re", "Im"])

        for i, fHz in enumerate(freqs):
            re = sdata[2*i]
            im = sdata[2*i + 1]
            w.writerow([fHz, re, im])

    sf.print_green(f"Saved: {local_csv}")

print("Final err:", sf.query_text(VNA, "SYST:ERR?"))