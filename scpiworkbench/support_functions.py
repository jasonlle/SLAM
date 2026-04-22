from termcolor import colored
import os
import time
import csv

def print_blue(msg):
    print(colored(msg,'blue'))

def print_yellow(msg):
    print(colored(msg,'yellow'))

def print_red(msg):
    print(colored(msg,'light_red'))

def print_green(msg):
    print(colored(msg,'light_green'))

def get_blue(msg):
    return colored(msg,'blue')

def get_yellow(msg):
    return colored(msg,'yellow')

def get_red(msg):
    return colored(msg,'light_red')

def get_green(msg):
    return colored(msg,'light_green')

def check_for_error(instrument,print_error=False):
    err = instrument.query("SYST:ERR?")
    if '+0,"No error"' not in err:
        if print_error:
            print_red(err)
        return True
    else:
        if print_error:
            print_green('\nDevice has no errors.')
        return False

def toggle_preset(instrument):
    instrument.write("SYST:PRES")
    instrument.write("DISP:WIND1:STATE ON")

if __name__ == '__main__':
    msg = 'TESTING'
    print_blue(msg)
    print_yellow(msg)
    print_red(msg)
    print_green(msg)

def scpi_get_error(inst) -> str:
    """Read one error from the SCPI error queue."""
    try:
        return inst.query("SYST:ERR?").strip()
    except Exception as e:
        return f"(could not read SYST:ERR?): {e}"

def scpi_clear_errors(inst, max_reads: int = 20):
    """Drain the error queue (optional)."""
    for _ in range(max_reads):
        e = scpi_get_error(inst)
        if e.startswith("+0") or e.startswith("0"):
            break

def read_ieee4882_block_visalib(inst) -> bytes:
    """
    Robust IEEE488.2 block read using VISA byte reads.
    Works even if the header arrives separately.
    """
    # Read '#' then one digit
    h = inst.read_bytes(2)  # b'#' + ndigit
    if h[0:1] != b'#':
        # Not a block; read any pending text to help debug
        try:
            rest = inst.read_raw()
        except Exception:
            rest = b""
        raise RuntimeError(f"Expected '#', got {h!r} rest={rest[:200]!r}")

    ndigits = int(h[1:2].decode("ascii"))
    if ndigits < 1 or ndigits > 9:
        raise RuntimeError(f"Invalid ndigits={ndigits}")

    # Read length field
    len_bytes = inst.read_bytes(ndigits)
    nbytes = int(len_bytes.decode("ascii"))

    # Read payload
    payload = inst.read_bytes(nbytes)
    return payload

def scpi_get_file(inst, remote_path: str) -> bytes:
    inst.write(f"MMEMory:TRAN? '{remote_path}'")
    data = read_ieee4882_block_visalib(inst)   # read the block right away
    # optional: check error after transfer
    # print("After TRAN err:", inst.query("SYST:ERR?").strip())
    return data

def scpi_save_then_pull_csv(inst, remote_csv_path: str, local_csv_path: str):
    """Pull a CSV file that exists on the instrument and save it on the PC."""
    file_bytes = scpi_get_file(inst, remote_csv_path)

    local_dir = os.path.dirname(local_csv_path)
    if local_dir:
        os.makedirs(local_dir, exist_ok=True)

    with open(local_csv_path, "wb") as f:
        f.write(file_bytes)

def query_any(inst, cmd: str) -> str:
    """
    Query that can handle either:
      - plain text (terminated), or
      - IEEE488.2 definite-length block (#<n><len><data>)
    Returns a decoded text string.
    """
    inst.write(cmd)

    raw = inst.read_raw()  # grab what arrives first
    if not raw:
        return ""

    # If it's a block, parse it and read the rest if needed
    if raw.startswith(b"#"):
        if len(raw) < 2:
            raw += inst.read_raw()
        nd = int(raw[1:2].decode("ascii"))
        while len(raw) < 2 + nd:
            raw += inst.read_raw()
        nbytes = int(raw[2:2+nd].decode("ascii"))
        payload = raw[2+nd:]

        while len(payload) < nbytes:
            payload += inst.read_raw()

        data = payload[:nbytes]
        return data.decode("ascii", errors="replace").strip()

    # Otherwise it's plain text
    return raw.decode("ascii", errors="replace").strip()

def query_block_or_text(inst, cmd: str) -> str:
    inst.write(cmd)

    first = inst.read_bytes(2)          # either "#<n>" or first 2 chars of text
    if first[0:1] != b"#":
        # plain text
        rest = inst.read_raw()
        return (first + rest).decode("ascii", errors="replace").strip()

    ndigits = int(first[1:2].decode("ascii"))
    len_bytes = inst.read_bytes(ndigits)
    nbytes = int(len_bytes.decode("ascii"))

    payload = inst.read_bytes(nbytes)
    return payload.decode("ascii", errors="replace").strip()

def save_active_trace_to_csv_ena(inst, local_csv_path: str, channel: int = 1):
    import os, csv

    inst.write("FORM:DATA ASC")

    x_str = query_block_or_text(inst, f"CALC{channel}:X?")
    freqs = [float(x) for x in x_str.split(",") if x]

    s_str = query_block_or_text(inst, f"CALC{channel}:DATA? SDATA")
    nums = [float(x) for x in s_str.split(",") if x]

    if len(nums) != 2 * len(freqs):
        raise RuntimeError(f"Length mismatch: freq={len(freqs)} sdata={len(nums)}")

    os.makedirs(os.path.dirname(local_csv_path), exist_ok=True)
    with open(local_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Frequency_Hz", "Re", "Im"])
        for i, fHz in enumerate(freqs):
            w.writerow([fHz, nums[2*i], nums[2*i + 1]])

def query_csv_numbers(inst, cmd: str):
    """
    Send a query that returns comma-separated ASCII floats (may not end with newline).
    Reads until no more data is immediately available, then parses floats.
    """
    inst.write(cmd)

    # Read first chunk
    data = inst.read_raw()

    # Keep draining quickly-available chunks (prevents leftover bytes -> -420)
    # Use a short temporary timeout to detect "no more data".
    old_timeout = inst.timeout
    try:
        inst.timeout = 200  # ms
        while True:
            try:
                more = inst.read_raw()
                if not more:
                    break
                data += more
            except Exception:
                break
    finally:
        inst.timeout = old_timeout

    text = data.decode("ascii", errors="replace").strip()
    # Some responses can include trailing commas/newlines; filter empties.
    return [float(x) for x in text.split(",") if x]

def query_text(inst, cmd: str) -> str:
    inst.write(cmd)
    data = inst.read_raw()
    # drain any extra quickly-available bytes
    old_timeout = inst.timeout
    try:
        inst.timeout = 200
        while True:
            try:
                more = inst.read_raw()
                if not more:
                    break
                data += more
            except Exception:
                break
    finally:
        inst.timeout = old_timeout
    return data.decode("ascii", errors="replace").strip()