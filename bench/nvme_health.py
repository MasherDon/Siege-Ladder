"""NVMe health: temperature sensors, thresholds, HCTM (host-controlled thermal management) and its throttling counters,
from the drive's SMART / Health log and Identify Controller data.

Reads through IOCTL_STORAGE_QUERY_PROPERTY (the channel Get-StorageReliabilityCounter uses): no Linux, no HWiNFO,
nothing written to the drive. Windows only.

    python bench/nvme_health.py           # the drive that holds SL_MODEL
    python bench/nvme_health.py 3         # a physical drive by number
    python bench/nvme_health.py E         # the drive behind a drive letter
"""

import ctypes
import ctypes.wintypes as wt
import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

IOCTL_STORAGE_QUERY_PROPERTY = 0x002D1400
ADAPTER_PROTOCOL, DEVICE_PROTOCOL = 49, 50          # StorageAdapter/DeviceProtocolSpecificProperty
NVME, IDENTIFY, LOG_PAGE, FEATURE = 3, 1, 2, 3       # ProtocolTypeNvme, NVMeDataType*

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.CreateFileW.restype = wt.HANDLE
k32.CreateFileW.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD, ctypes.c_void_p, wt.DWORD, wt.DWORD, wt.HANDLE]
k32.DeviceIoControl.restype = wt.BOOL
k32.DeviceIoControl.argtypes = [wt.HANDLE, wt.DWORD, ctypes.c_void_p, wt.DWORD, ctypes.c_void_p, wt.DWORD,
                                ctypes.POINTER(wt.DWORD), ctypes.c_void_p]
k32.CloseHandle.argtypes = [wt.HANDLE]


def query(h, prop, dtype, value, length, sub=0):
    """-> (data, FixedProtocolReturnData). The request: PropertyId, QueryType, STORAGE_PROTOCOL_SPECIFIC_DATA, buffer.
    Get Features without a data buffer (offset and length 0) answers in FixedProtocolReturnData (Dword0)."""
    psd = struct.pack("<10I", NVME, dtype, value, sub, 40 if length else 0, length, 0, 0, 0, 0)
    buf = ctypes.create_string_buffer(struct.pack("<II", prop, 0) + psd + bytes(length))
    got = wt.DWORD(0)
    if not k32.DeviceIoControl(h, IOCTL_STORAGE_QUERY_PROPERTY, buf, len(buf), buf, len(buf), ctypes.byref(got), None):
        raise OSError(ctypes.get_last_error(), f"query {dtype}/{value:#x} failed")
    off, ln, fixed = struct.unpack_from("<10I", buf.raw, 8)[4:7]
    return buf.raw[8 + off:8 + off + ln], fixed


def c(kelvin):
    return None if kelvin == 0 else kelvin - 273


def model_letter():
    """The drive letter of SL_MODEL, or None."""
    return config.MODEL.resolve().drive.rstrip(":") if config.MODEL else None


def disk_of(letter=None):
    letter = letter or model_letter()
    if not letter:
        sys.exit(f"give a disk number or a drive letter, or set SL_MODEL (environment or {config.FILE})")
    return int(subprocess.run(["powershell", "-NoProfile", "-Command", f"(Get-Partition -DriveLetter {letter}).DiskNumber"],
                              capture_output=True, text=True).stdout.strip())


def open_disk(disk):
    h = k32.CreateFileW(rf"\\.\PhysicalDrive{disk}", 0, 3, None, 3, 0, None)   # no data access
    if h in (None, wt.HANDLE(-1).value):
        raise OSError(ctypes.get_last_error(), f"cannot open PhysicalDrive{disk}")
    return h


def health(disk):
    """-> {composite, sensors, HCTM counters} from the SMART / Health log: for measurements that track throttling."""
    h = open_disk(disk)
    try:
        log, _ = query(h, DEVICE_PROTOCOL, LOG_PAGE, 2, 512)
    finally:
        k32.CloseHandle(h)
    t1n, t2n, t1s, t2s = struct.unpack_from("<4I", log, 216)
    return dict(t_comp=c(struct.unpack_from("<H", log, 1)[0]),
                t_sens=[c(k) for k in struct.unpack_from("<8H", log, 200) if k],
                tmt1_n=t1n, tmt1_s=t1s, tmt2_n=t2n, tmt2_s=t2s)


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    disk = int(arg) if arg and arg.isdigit() else disk_of(arg)
    h = open_disk(disk)
    try:
        idc, _ = query(h, ADAPTER_PROTOCOL, IDENTIFY, 1, 4096)
        model = idc[24:64].decode("ascii", "replace").strip()
        fw = idc[64:72].decode("ascii", "replace").strip()
        wctemp, cctemp = struct.unpack_from("<HH", idc, 266)
        hctma, mntmt, mxtmt = struct.unpack_from("<HHH", idc, 322)
        print(f"PhysicalDrive{disk}: {model}, firmware {fw}")
        print(f"  thresholds: warning {c(wctemp)} °C, critical {c(cctemp)} °C")
        print(f"  HCTM: {'yes' if hctma & 1 else 'no'}; TMT range {c(mntmt)}...{c(mxtmt)} °C")
        npss = idc[263]
        states = []
        for i in range(npss + 1):
            mp, _, flags = struct.unpack_from("<HBB", idc, 2048 + 32 * i)
            watts = mp * (0.0001 if flags & 1 else 0.01)
            states.append(f"PS{i} {watts:.2f} W{' (non-operational)' if flags & 2 else ''}")
        print("  power states: " + ", ".join(states))

        log, _ = query(h, DEVICE_PROTOCOL, LOG_PAGE, 2, 512)
        comp = struct.unpack_from("<H", log, 1)[0]
        warn_min, crit_min = struct.unpack_from("<II", log, 192)
        sensors = struct.unpack_from("<8H", log, 200)
        t1n, t2n, t1s, t2s = struct.unpack_from("<4I", log, 216)
        busy_min = int.from_bytes(log[96:112], "little")
        read_tb = int.from_bytes(log[32:48], "little") * 512000 / 1e12
        print(f"  now: composite {c(comp)} °C; sensors " +
              ", ".join(f"{i + 1}: {c(k)} °C" for i, k in enumerate(sensors) if k))
        print(f"  above the warning threshold {warn_min} min, above the critical one {crit_min} min")
        print(f"  HCTM transitions: TMT1 {t1n} times / {t1s} s, TMT2 {t2n} times / {t2s} s")
        print(f"  read in total {read_tb:.1f} TB, controller busy {busy_min} min")

        for prop in (DEVICE_PROTOCOL, ADAPTER_PROTOCOL):
            try:
                _, dw0 = query(h, prop, FEATURE, 0x10, 0)
                print(f"  HCTM now: TMT1 {c(dw0 >> 16)} °C, TMT2 {c(dw0 & 0xFFFF)} °C (None - not set)")
                break
            except OSError as e:
                print(f"  HCTM now (property {prop}): cannot read ({e})")
    finally:
        k32.CloseHandle(h)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
