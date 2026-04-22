import pyvisa
import support_functions as sf
import time
import os
import csv
from datetime import datetime

addr = 'TCPIP0::192.168.1.10::inst0::INSTR'
VNA_model = 'E5071C'

low_freq = 902e6
upper_freq = 928e6

s_params_list = ['21']

reset_to_instrument_preset = False

manual_step_size = True
step_freq = 15e6

pc_dir = r"C:\Users\ericd\Desktop\dataset\\"

# Set to an integer if you want a fixed number of captures
# Set to 0 to run until you stop with Ctrl+C
num_captures = 30

print('\nAttempting to connect to: ' + sf.get_yellow(addr))

rm = pyvisa.ResourceManager()
VNA = rm.open_resource(addr)

VNA.write_termination = "\n"
VNA.read_termination = None
VNA.timeout = 60000

resp = sf.query_text(VNA, "*IDN?")
VNA.write(f"CALC1:PAR:COUN {len(s_params_list)}")

if VNA_model in resp:
    sf.print_green('Successfully connected to Keysight VNA!\n')
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

# Define traces once
for trace_idx, s_param in enumerate(s_params_list, start=1):
    meas = f"S{s_param}"
    VNA.write(f"CALC1:PAR{trace_idx}:DEF {meas}")
    VNA.write(f"DISP:WIND1:TRAC{trace_idx}:FEED '{meas}'")

os.makedirs(pc_dir, exist_ok=True)

capture_count = 0
start_time = time.time()

try:
    while True:
        if num_captures is not None and capture_count >= num_captures:
            break

        sweep_start = time.time()

        # Use microseconds so filenames never collide
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        for trace_idx, s_param in enumerate(s_params_list, start=1):
            meas = f"S{s_param}"

            VNA.write("DISP:WIND1:ACT")
            VNA.write(f"CALC1:PAR{trace_idx}:SEL")

            # Trigger one sweep and wait until done
            VNA.write("INIT1:IMM")
            sf.query_text(VNA, "*OPC?")

            freqs = sf.query_csv_numbers(VNA, "SENS1:FREQ:DATA?")
            sdata = sf.query_csv_numbers(VNA, "CALC1:DATA:SDAT?")

            if len(sdata) != 2 * len(freqs):
                raise RuntimeError(f"Mismatch: freqs={len(freqs)} sdata={len(sdata)}")

            local_csv = os.path.join(pc_dir, f"{meas}_{timestamp}.csv")

            with open(local_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["Frequency_Hz", "Re", "Im"])

                for i, fHz in enumerate(freqs):
                    re = sdata[2 * i]
                    im = sdata[2 * i + 1]
                    w.writerow([fHz, re, im])

            sf.print_green(f"Saved: {local_csv}")

        capture_count += 1
        sweep_elapsed = time.time() - sweep_start
        print(f"Capture {capture_count} done in {sweep_elapsed:.3f} s")

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    total_time = time.time() - start_time
    if total_time > 0:
        print(f"Average rate: {capture_count / total_time:.3f} captures/sec")

    print("Final err:", sf.query_text(VNA, "SYST:ERR?"))
    VNA.close()
    rm.close()