import pyvisa, time

addr = "TCPIP0::192.168.1.10::inst0::INSTR"

rm = pyvisa.ResourceManager()
VNA = rm.open_resource(addr)
VNA.write_termination = "\n"
VNA.read_termination = "\n"
VNA.timeout = 3000  # 3s so nothing "freezes"

print("IDN:", VNA.query("*IDN?").strip())

def try_query(label, cmd):
    print(f"\n[{label}] {cmd}")
    try:
        r = VNA.query(cmd)
        print("OK head:", r[:80])
    except Exception as e:
        print("FAIL:", type(e).__name__, e)
    try:
        print("ERRQ:", VNA.query("SYST:ERR?").strip())
    except Exception as e:
        print("ERRQ read failed:", e)

# Basic setup
VNA.write("*CLS")
VNA.write("SENS1:FREQ:STAR 3e8")
VNA.write("SENS1:FREQ:STOP 6e9")
VNA.write("CALC1:PAR:DEF S11")
VNA.write("CALC1:PAR:SEL 'S11'")
VNA.write("INIT1:CONT OFF")
VNA.write("INIT1:IMM")
time.sleep(0.2)

# Try stimulus axis queries
try_query("freq1", "SENS1:FREQ:DATA?")
try_query("freq2", "SENS1:X?")          # some personalities use this
try_query("freq3", "CALC1:X?")          # you said this hangs (will time out fast)

# Try data queries in several dialects
try_query("data1", "CALC1:DATA?")
try_query("data2", "CALC1:DATA? FDATA")
try_query("data3", "CALC1:DATA? SDATA")
try_query("data4", "CALC1:DATA:FDAT?")
try_query("data5", "CALC1:DATA:SDAT?")