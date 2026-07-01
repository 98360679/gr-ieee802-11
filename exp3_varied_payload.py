"""
exp3_varied_payload.py — GNU Radio Embedded Python Block: varied WiFi payload.

Replaces wifi_tx.grc's FIXED all-'x' Message-Strobe payload
    pmt.intern("".join("x" for i in range(pdu_length)))
with a FRESH RANDOM payload on every frame, so the PA is driven across many operating
points and the hardware fingerprint is fully exposed (fixes the content-starvation that
collapses fingerprint separability, e.g. 0.52/0.77 -> 0.95/0.99).

WIRING in wifi_tx.grc (insert this block between the Message Strobe and the MAC):
    Message Strobe [strobe] ──▶ [trig] exp3_varied_payload [out] ──▶ [app in] ieee802_11_mac_0
The Message Strobe now only *triggers* this block (its own msg content is ignored); this
block emits the actual, varying payload. Set the block's `pdu_length` = the flowgraph's
`pdu_length` variable. Keep everything else (gain, interval) identical across all devices
and both runs.

To use as an Embedded Python Block: in GRC add "Embedded Python Block", open its editor,
and paste this file's contents (class name must stay `blk`).
"""
import pmt
import random
import string
from gnuradio import gr


class blk(gr.basic_block):
    """Emit a fresh random WiFi payload on each strobe trigger (content-rich enrollment)."""

    def __init__(self, pdu_length=1500, seed=0):
        gr.basic_block.__init__(self, name='exp3_varied_payload', in_sig=[], out_sig=[])
        self.pdu_length = int(pdu_length)
        self.rng = random.Random(int(seed))
        self.alphabet = string.ascii_letters + string.digits   # printable => safe pmt symbol
        self.message_port_register_in(pmt.intern('trig'))
        self.message_port_register_out(pmt.intern('out'))
        self.set_msg_handler(pmt.intern('trig'), self.handle)

    def handle(self, _msg):
        # fresh random payload each strobe -> varied OFDM symbols -> full fingerprint exposure
        payload = ''.join(self.rng.choice(self.alphabet) for _ in range(self.pdu_length))
        self.message_port_pub(pmt.intern('out'), pmt.intern(payload))
