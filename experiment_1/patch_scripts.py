#!/usr/bin/env python3
"""
patch_scripts.py — Adds device/serial prompts to wifi_tx_perturb.py and wifi_rx_perturb.py.
Run once. After that, just run the scripts as normal — they will prompt you.

Usage:
    python3 patch_scripts.py
"""

import os, re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TX_SCRIPT  = os.path.join(SCRIPT_DIR, "wifi_tx_perturb.py")
RX_SCRIPT  = os.path.join(SCRIPT_DIR, "wifi_rx_perturb.py")

# ── Patch wifi_tx_perturb.py ─────────────────────────────────────────────────
TX_OLD = 'if __name__ == \'__main__\':\n    main()'

TX_NEW = '''if __name__ == '__main__':
    serial = input("Enter TX serial number (e.g. 3259376): ").strip()
    device_id = input("Enter device ID (0-6): ").strip()
    import wifi_tx_perturb as _m
    # Patch serial
    _m.wifi_tx.__init__.__globals__
    # We patch via sys.argv is messy — instead patch the class before instantiation
    _orig_init = wifi_tx.__init__
    def _patched_init(self):
        _orig_init(self)
        # Patch serial
        self.uhd_usrp_sink_0.set_subdev_spec("", 0)
    import sys, os
    # Simplest: patch the source before importing
    main(serial=serial, device_id=device_id)
'''

# The above is too complex. Do it cleanly by modifying main() to accept params
# and adding prompt logic at __main__

TX_MAIN_OLD = '''def main(top_block_cls=wifi_tx, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()'''

TX_MAIN_NEW = '''def main(top_block_cls=wifi_tx, options=None, serial=None, device_id=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls(serial=serial, device_id=device_id)'''

TX_INIT_OLD = '''    def __init__(self):
        gr.top_block.__init__(self, "Wifi Tx", catch_exceptions=True)
        Qt.QWidget.__init__(self)'''

TX_INIT_NEW = '''    def __init__(self, serial=None, device_id=None):
        gr.top_block.__init__(self, "Wifi Tx", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        if serial is None:
            serial = input("Enter TX serial number (e.g. 3259376): ").strip()
        if device_id is None:
            device_id = input("Enter device ID (0-6): ").strip()
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", f"device_{device_id}")
        os.makedirs(data_dir, exist_ok=True)
        tx_bin_path = os.path.join(data_dir, "tx.bin")'''

TX_SERIAL_OLD = '        "serial=3259376",'
TX_SERIAL_NEW = '        f"serial={serial}",'

TX_TXPATH_OLD = '        tx_path = "/home/misty/research/adversarial/data/tx.bin"'
TX_TXPATH_NEW = '        tx_path = tx_bin_path'

TX_SINK_OLD = '            gr.sizeof_gr_complex, "/home/misty/research/adversarial/data/tx.bin", False)'
TX_SINK_NEW = '            gr.sizeof_gr_complex, tx_bin_path, False)'

TX_MAIN_CALL_OLD = "if __name__ == '__main__':\n    main()"
TX_MAIN_CALL_NEW = """if __name__ == '__main__':
    serial    = input("Enter TX serial number (e.g. 3259376): ").strip()
    device_id = input("Enter device ID (0-6): ").strip()
    main(serial=serial, device_id=device_id)"""

# ── Patch wifi_rx_perturb.py ─────────────────────────────────────────────────
RX_INIT_OLD = '''class wifi_rx(gr.top_block, Qt.QWidget):

    def __init__(self):'''

RX_INIT_NEW = '''class wifi_rx(gr.top_block, Qt.QWidget):

    def __init__(self, device_id=None):'''

RX_VARS_OLD = "        self.samp_rate = samp_rate = 5e6"
RX_VARS_NEW = """        if device_id is None:
            device_id = input("Enter device ID (0-6): ").strip()
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", f"device_{device_id}")
        os.makedirs(data_dir, exist_ok=True)
        self.samp_rate = samp_rate = 5e6"""

RX_MAIN_OLD = '''def main(top_block_cls=wifi_rx, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()'''

RX_MAIN_NEW = '''def main(top_block_cls=wifi_rx, options=None, device_id=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls(device_id=device_id)'''

RX_AF_OLD    = "'/home/nghoselab/research/adversarial/data/af.bin'"
RX_AFTER_OLD = "'/home/nghoselab/research/adversarial/data/after.bin'"
RX_EQU_OLD   = "'/home/nghoselab/research/adversarial/data/equ.bin'"

RX_AF_NEW    = "os.path.join(data_dir, 'af.bin')"
RX_AFTER_NEW = "os.path.join(data_dir, 'after.bin')"
RX_EQU_NEW   = "os.path.join(data_dir, 'equ.bin')"

RX_MAIN_CALL_OLD = "if __name__ == '__main__':\n    main()"
RX_MAIN_CALL_NEW = """if __name__ == '__main__':
    device_id = input("Enter device ID (0-6): ").strip()
    main(device_id=device_id)"""


def patch(script_path, replacements):
    with open(script_path) as f:
        content = f.read()
    for old, new in replacements.items():
        if old in content:
            content = content.replace(old, new)
            print(f"  [OK] {old[:60].strip()}")
        else:
            print(f"  [WARN] Not found: {old[:60].strip()}")
    with open(script_path, 'w') as f:
        f.write(content)


print("Patching wifi_tx_perturb.py...")
patch(TX_SCRIPT, {
    TX_INIT_OLD:        TX_INIT_NEW,
    TX_MAIN_OLD:        TX_MAIN_NEW,
    TX_SERIAL_OLD:      TX_SERIAL_NEW,
    TX_SINK_OLD:        TX_SINK_NEW,
    TX_TXPATH_OLD:      TX_TXPATH_NEW,
    TX_MAIN_CALL_OLD:   TX_MAIN_CALL_NEW,
})

print("\nPatching wifi_rx_perturb.py...")
patch(RX_SCRIPT, {
    RX_INIT_OLD:        RX_INIT_NEW,
    RX_VARS_OLD:        RX_VARS_NEW,
    RX_MAIN_OLD:        RX_MAIN_NEW,
    RX_AF_OLD:          RX_AF_NEW,
    RX_AFTER_OLD:       RX_AFTER_NEW,
    RX_EQU_OLD:         RX_EQU_NEW,
    RX_MAIN_CALL_OLD:   RX_MAIN_CALL_NEW,
})

print("\nDone. Now just run the scripts as normal:")
print("  TX: python3 wifi_tx_perturb.py  → prompts for serial and device ID")
print("  RX: python3 wifi_rx_perturb.py  → prompts for device ID")
