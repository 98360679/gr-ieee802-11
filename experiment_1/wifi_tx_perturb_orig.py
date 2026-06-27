

from PyQt5 import Qt
from gnuradio import qtgui
import os
import sys
sys.path.append(os.environ.get('GRC_HIER_PATH', os.path.expanduser('~/.grc_gnuradio')))

from PyQt5 import QtCore
from PyQt5.QtCore import QObject, pyqtSlot
from gnuradio import blocks
import pmt
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import network
from gnuradio import uhd
import time
from wifi_phy_hier import wifi_phy_hier  # grc-generated hier_block
import foo
import ieee802_11
import itertools




class wifi_tx(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Wifi Tx", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Wifi Tx")
        qtgui.util.check_set_qss()

        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)

        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)



        self.settings = Qt.QSettings("GNU Radio", "wifi_tx")
        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)

        # Variables
        self.tx_gain = 1.0
        self.samp_rate = 5e6
        self.pdu_length = 500
        self.out_buf_size = 96000
        self.lo_offset = 0
        self.interval = 300  # milliseconds
        self.freq = 2450000000
        self.encoding = 0
        self.sample_counter = 0
        self.samples_per_frame = int(self.samp_rate * self.interval / 1000.0)

        # Frame ID generator
        self.frame_id_gen = itertools.count()

        # Blocks
        self.wifi_phy_hier_0 = wifi_phy_hier(
            bandwidth=self.samp_rate,
            chan_est=ieee802_11.LS,
            encoding=ieee802_11.Encoding(self.encoding),
            frequency=self.freq,
            sensitivity=0.56,
        )

        self.uhd_usrp_sink_0 = uhd.usrp_sink(
            "serial=3259373",
            uhd.stream_args(cpu_format="fc32", args='', channels=[0]),
            "packet_len"
        )
        self.uhd_usrp_sink_0.set_samp_rate(self.samp_rate)
        self.uhd_usrp_sink_0.set_time_unknown_pps(uhd.time_spec(0))
        self.uhd_usrp_sink_0.set_center_freq(
            uhd.tune_request(self.freq, rf_freq=self.freq - self.lo_offset,
                             rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 0)
        self.uhd_usrp_sink_0.set_normalized_gain(self.tx_gain, 0)

        self.network_socket_pdu_0 = network.socket_pdu('TCP_SERVER', '', '52001', 10000, False)
        self.ieee802_11_mac_0 = ieee802_11.mac(
            [0x23]*6, [0x42]*6, [0xff]*6)

        self.foo_packet_pad2_0 = foo.packet_pad2(False, False, 0.01, 100, 1000)
        self.foo_packet_pad2_0.set_min_output_buffer(self.out_buf_size)

        self.blocks_vector_source_x_0 = blocks.vector_source_c(
            [1+0j, 0.998+0.063j, 0.992+0.125j, 0.982+0.187j], False, 1, [])
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_cc(0.6)
        self.blocks_multiply_const_vxx_0.set_min_output_buffer(100000)

        self.blocks_file_sink_0 = blocks.file_sink(
            gr.sizeof_gr_complex, "/home/misty/Desktop/Code-Data/tx.bin", False)
        self.blocks_file_sink_0.set_unbuffered(True)
        self.msg_connect((self.network_socket_pdu_0, 'pdus'), (self.ieee802_11_mac_0, 'app in'))
        self.msg_connect((self.ieee802_11_mac_0, 'phy out'), (self.wifi_phy_hier_0, 'mac_in'))
        self.connect((self.blocks_vector_source_x_0, 0), (self.wifi_phy_hier_0, 0))
        self.connect((self.wifi_phy_hier_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.wifi_phy_hier_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.foo_packet_pad2_0, 0))
        self.connect((self.foo_packet_pad2_0, 0), (self.uhd_usrp_sink_0, 0))




        # Start message-sending timer
        self.msg_timer = QtCore.QTimer()
        self.msg_timer.timeout.connect(self.send_dynamic_frame)
        self.msg_timer.start(self.interval)


    @pyqtSlot()
    def send_dynamic_frame(self):

        tx_path = "/home/misty/Desktop/Code-Data/tx.bin"
        try:
            current_size = os.path.getsize(tx_path)
            current_samples = current_size // 8  # complex64 = 8 bytes (2 × 32-bit floats)
        except FileNotFoundError:
            current_samples = 0

        frame_id_int = next(self.frame_id_gen)
        frame_id_str = f"{frame_id_int:04d}"

        print(f"frame ID: {frame_id_str} @ sample {current_samples}")

        pdu = self.generate_payload_with_id(self.pdu_length, frame_id_int)
        self.ieee802_11_mac_0.to_basic_block()._post(pmt.intern('app in'), pdu)



    def generate_payload_with_id(self, pdu_length, frame_id):
        frame_id_str = f"{frame_id:04d}"
        payload = frame_id_str + ''.join(str(i % 10) for i in range(pdu_length - 4))
        payload_bytes = payload.encode('utf-8')
        pdu = pmt.cons(pmt.PMT_NIL, pmt.init_u8vector(len(payload_bytes), bytearray(payload_bytes)))
        return pdu

   



    def closeEvent(self, event):
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()
        event.accept()

    # GUI-controlled setters/getters
    def get_tx_gain(self): return self.tx_gain
    def set_tx_gain(self, tx_gain):
        self.tx_gain = tx_gain
        self.uhd_usrp_sink_0.set_normalized_gain(tx_gain, 0)

    def get_samp_rate(self): return self.samp_rate
    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.uhd_usrp_sink_0.set_samp_rate(samp_rate)
        self.wifi_phy_hier_0.set_bandwidth(samp_rate)

    def get_pdu_length(self): return self.pdu_length
    def set_pdu_length(self, pdu_length):
        self.pdu_length = pdu_length

    def get_lo_offset(self): return self.lo_offset
    def set_lo_offset(self, lo_offset):
        self.lo_offset = lo_offset
        self.uhd_usrp_sink_0.set_center_freq(
            uhd.tune_request(self.freq, rf_freq=self.freq - lo_offset,
                             rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 0)

    def get_interval(self): return self.interval
    def set_interval(self, interval):
        self.interval = interval
        self.msg_timer.setInterval(self.interval)

    def get_freq(self): return self.freq
    def set_freq(self, freq):
        self.freq = freq
        self.uhd_usrp_sink_0.set_center_freq(
            uhd.tune_request(freq, rf_freq=freq - self.lo_offset,
                             rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 0)
        self.wifi_phy_hier_0.set_frequency(freq)

    def get_encoding(self): return self.encoding
    def set_encoding(self, encoding):
        self.encoding = encoding
        self.wifi_phy_hier_0.set_encoding(ieee802_11.Encoding(encoding))






def main(top_block_cls=wifi_tx, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
