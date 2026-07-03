"""
exp3_varied_payload.py — GNU Radio Embedded Python Block: FRID-tagged VARIED WiFi payload.

Replaces wifi_tx.grc's FIXED all-'x' Message-Strobe payload with a payload that is both
(a) VARIED (fresh body every frame -> drives the PA across many operating points ->
    exposes the hardware fingerprint; fixes content starvation 0.52/0.77 -> 0.95/0.99), and
(b) BER-COMPUTABLE (each frame carries FRID + frame_id, and the body is a DETERMINISTIC
    function of frame_id, so the analysis side can regenerate the exact transmitted bits).

MSDU layout (identical header to exp3_wifi_tx.py, only the body changes fixed->varied):
    bytes 0..3 : b'FRID' magic
    bytes 4..7 : uint32 big-endian frame_id  (sequential, starts at 0)
    bytes 8..N : varied_body(frame_id)        (deterministic per frame_id)

Because the body is seeded ONLY by frame_id, every device and every run transmits the SAME
content for a given frame_id -> content is varied AND identical across devices (separability
reflects hardware, not content), and BER can be measured by regenerating varied_body(id).

BER (regeneration method — the mode-reference in exp3_ber_eval does NOT work for varied fill):
    for each decoded frame: read frame_id, expected = b'FRID'+uint32_be(id)+varied_body(id),
    bit_errors = popcount(decoded_MSDU XOR expected); BER = sum(bit_errors)/sum(bits).

WIRING (wifi_tx_varied.grc already does this): Message Strobe [strobe] -> [trig] this block
[out] -> [app in] ieee802_11_mac_0. Set pdu_length = flowgraph's pdu_length (MSDU bytes).
"""
import pmt
import random
import struct
import itertools
from gnuradio import gr

MAGIC = b'FRID'


def varied_body(frame_id, length):
    """Deterministic per-frame body — MUST stay identical here and in the BER tool."""
    rng = random.Random(int(frame_id))
    return bytes(rng.getrandbits(8) for _ in range(int(length)))


def expected_msdu(frame_id, pdu_length):
    """The full transmitted MSDU for a frame_id (ground truth for BER)."""
    return MAGIC + struct.pack('>I', int(frame_id) & 0xFFFFFFFF) + varied_body(frame_id, pdu_length - 8)


class blk(gr.basic_block):
    """Emit a FRID-tagged, frame_id-seeded varied payload on each strobe trigger."""

    def __init__(self, pdu_length=500):
        gr.basic_block.__init__(self, name='exp3_varied_payload', in_sig=[], out_sig=[])
        self.pdu_length = int(pdu_length)
        self._ids = itertools.count()
        self.message_port_register_in(pmt.intern('trig'))
        self.message_port_register_out(pmt.intern('out'))
        self.set_msg_handler(pmt.intern('trig'), self.handle)

    def handle(self, _msg):
        fid = next(self._ids) & 0xFFFFFFFF
        payload = expected_msdu(fid, self.pdu_length)          # FRID + id + varied_body(id)
        pdu = pmt.cons(pmt.PMT_NIL, pmt.init_u8vector(len(payload), list(payload)))
        self.message_port_pub(pmt.intern('out'), pdu)
