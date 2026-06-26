"""
link_stats — live decode-rate / loss readout for the FRID closed loop
─────────────────────────────────────────────────────────────────────
Drop this into wifi_rx.grc as a GRC **Python Block** (Embedded Python Block),
and wire  decode_mac 'out'  ->  this block's 'in'  (same source as frame_logger).

Why decode-rate, not BER: decode_mac only emits CRC-VALID frames on 'out', and a
CRC-valid frame is error-free — so live BER off this port is always ~0 and
useless. Instead we exploit that the TX stamps SEQUENTIAL frame_ids (FRID): the
fraction of the transmitted id-range we actually decode is the real link-quality
signal. Tune rx_gain to MAXIMIZE the "decode-rate now" number.

Prints every `report_sec`:
  decode-rate now  : decoded / id-advance over the LAST interval  (the tuning knob)
  cum              : unique decoded / full id span               (overall)
  fps              : CRC-valid frames per second
  ambient-skip     : non-FRID frames seen (other WiFi / false syncs) — high = noisy

GRC: id `link_stats`, one message input port 'in', no outputs. Param: report_sec.
"""
import time
import struct
import pmt
from gnuradio import gr


class blk(gr.basic_block):
    MAC_HDR_LEN = 24            # 802.11 data-frame header before the MSDU
    MAGIC = b'FRID'            # MSDU marker: b'FRID' + uint32_be(frame_id) + 'x'*fill

    def __init__(self, report_sec=2.0):
        gr.basic_block.__init__(self, name='link_stats', in_sig=[], out_sig=[])
        self.report_sec = float(report_sec)
        self.ids = set()           # unique frame_ids decoded
        self.n = 0                 # total CRC-valid FRID frames
        self.skipped = 0           # non-FRID frames (ambient / false sync)
        self.id_min = None
        self.id_max = None
        # interval bookkeeping for the "now" rate
        self.t_last = time.time()
        self.n_last = 0
        self.id_max_last = None
        self.message_port_register_in(pmt.intern('in'))
        self.set_msg_handler(pmt.intern('in'), self.handle)

    def handle(self, msg):
        try:
            data = bytes(pmt.u8vector_elements(pmt.cdr(msg)))
        except Exception:
            return
        msdu = data[self.MAC_HDR_LEN:]
        if msdu[:4] != self.MAGIC or len(msdu) < 8:
            self.skipped += 1          # not ours: ambient WiFi or a false decode
            return
        fid = struct.unpack('>I', msdu[4:8])[0]
        self.ids.add(fid)
        self.n += 1
        if self.id_min is None or fid < self.id_min:
            self.id_min = fid
        if self.id_max is None or fid > self.id_max:
            self.id_max = fid

        now = time.time()
        if now - self.t_last >= self.report_sec and self.id_max is not None:
            # recent interval: how many ids advanced vs how many we decoded
            base = self.id_max_last if self.id_max_last is not None else self.id_min
            d_span = self.id_max - base
            d_dec = self.n - self.n_last
            recent = min(100.0, 100.0 * d_dec / d_span) if d_span > 0 else 0.0
            # cumulative
            span = self.id_max - self.id_min + 1
            cum = 100.0 * len(self.ids) / span if span else 0.0
            fps = d_dec / (now - self.t_last)
            print('[link_stats] decode-rate now %5.1f%%   cum %5.1f%% (%d/%d)   '
                  '%5.1f fps   ambient-skip %d'
                  % (recent, cum, len(self.ids), span, fps, self.skipped))
            self.t_last = now
            self.n_last = self.n
            self.id_max_last = self.id_max
