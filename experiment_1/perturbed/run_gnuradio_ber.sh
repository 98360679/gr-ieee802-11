#!/bin/bash
# Feed perturbed IQ files through GNU Radio wifi_rx for BER measurement
# Prerequisites:
#   1. wifi_rx_file.grc with File Source (not USRP)
#   2. CRC-patched decode_mac.cc (passes corrupted packets)
#   3. BER counter embedded Python block
#
# Usage: bash run_gnuradio_ber.sh

DIR="/home/nghoselab/research/adversarial/perturbed"
RESULTS="$DIR/gnuradio_ber_results.csv"
WIFI_RX="/home/nghoselab/gr-ieee802-11/examples/wifi_rx_file.py"

echo "attack,epsilon,file,ber,per,total_pkts,good_pkts,bad_pkts" > $RESULTS

echo "=== Clean Baseline ==="
# Manually run: python3 $WIFI_RX --input $DIR/device1_session1_clean.bin
# Record BER from output


echo "=== FGSM ==="
echo "--- fgsm eps=0.0005 ---"
# python3 $WIFI_RX --input "$DIR/device1_session1_fgsm_eps0.0005.bin"
# Record: fgsm,0.0005,device1_session1_fgsm_eps0.0005.bin,<ber>,<per>,<total>,<good>,<bad>

echo "--- fgsm eps=0.001 ---"
# python3 $WIFI_RX --input "$DIR/device1_session1_fgsm_eps0.0010.bin"
# Record: fgsm,0.001,device1_session1_fgsm_eps0.0010.bin,<ber>,<per>,<total>,<good>,<bad>

echo "--- fgsm eps=0.002 ---"
# python3 $WIFI_RX --input "$DIR/device1_session1_fgsm_eps0.0020.bin"
# Record: fgsm,0.002,device1_session1_fgsm_eps0.0020.bin,<ber>,<per>,<total>,<good>,<bad>

echo "--- fgsm eps=0.005 ---"
# python3 $WIFI_RX --input "$DIR/device1_session1_fgsm_eps0.0050.bin"
# Record: fgsm,0.005,device1_session1_fgsm_eps0.0050.bin,<ber>,<per>,<total>,<good>,<bad>

echo "--- fgsm eps=0.01 ---"
# python3 $WIFI_RX --input "$DIR/device1_session1_fgsm_eps0.0100.bin"
# Record: fgsm,0.01,device1_session1_fgsm_eps0.0100.bin,<ber>,<per>,<total>,<good>,<bad>

echo "--- fgsm eps=0.02 ---"
# python3 $WIFI_RX --input "$DIR/device1_session1_fgsm_eps0.0200.bin"
# Record: fgsm,0.02,device1_session1_fgsm_eps0.0200.bin,<ber>,<per>,<total>,<good>,<bad>

echo "--- fgsm eps=0.05 ---"
# python3 $WIFI_RX --input "$DIR/device1_session1_fgsm_eps0.0500.bin"
# Record: fgsm,0.05,device1_session1_fgsm_eps0.0500.bin,<ber>,<per>,<total>,<good>,<bad>

echo "--- fgsm eps=0.07 ---"
# python3 $WIFI_RX --input "$DIR/device1_session1_fgsm_eps0.0700.bin"
# Record: fgsm,0.07,device1_session1_fgsm_eps0.0700.bin,<ber>,<per>,<total>,<good>,<bad>

echo "--- fgsm eps=0.09 ---"
# python3 $WIFI_RX --input "$DIR/device1_session1_fgsm_eps0.0900.bin"
# Record: fgsm,0.09,device1_session1_fgsm_eps0.0900.bin,<ber>,<per>,<total>,<good>,<bad>

echo "--- fgsm eps=0.1 ---"
# python3 $WIFI_RX --input "$DIR/device1_session1_fgsm_eps0.1000.bin"
# Record: fgsm,0.1,device1_session1_fgsm_eps0.1000.bin,<ber>,<per>,<total>,<good>,<bad>


echo "=== PGD ==="
echo "--- pgd eps=0.0005 ---"
# python3 $WIFI_RX --input "$DIR/device1_session1_pgd_eps0.0005.bin"
# Record: pgd,0.0005,device1_session1_pgd_eps0.0005.bin,<ber>,<per>,<total>,<good>,<bad>

echo "--- pgd eps=0.001 ---"
# python3 $WIFI_RX --input "$DIR/device1_session1_pgd_eps0.0010.bin"
# Record: pgd,0.001,device1_session1_pgd_eps0.0010.bin,<ber>,<per>,<total>,<good>,<bad>

echo "--- pgd eps=0.002 ---"
# python3 $WIFI_RX --input "$DIR/device1_session1_pgd_eps0.0020.bin"
# Record: pgd,0.002,device1_session1_pgd_eps0.0020.bin,<ber>,<per>,<total>,<good>,<bad>

echo "--- pgd eps=0.005 ---"
# python3 $WIFI_RX --input "$DIR/device1_session1_pgd_eps0.0050.bin"
# Record: pgd,0.005,device1_session1_pgd_eps0.0050.bin,<ber>,<per>,<total>,<good>,<bad>

echo "--- pgd eps=0.01 ---"
# python3 $WIFI_RX --input "$DIR/device1_session1_pgd_eps0.0100.bin"
# Record: pgd,0.01,device1_session1_pgd_eps0.0100.bin,<ber>,<per>,<total>,<good>,<bad>

echo "--- pgd eps=0.02 ---"
# python3 $WIFI_RX --input "$DIR/device1_session1_pgd_eps0.0200.bin"
# Record: pgd,0.02,device1_session1_pgd_eps0.0200.bin,<ber>,<per>,<total>,<good>,<bad>

echo "--- pgd eps=0.05 ---"
# python3 $WIFI_RX --input "$DIR/device1_session1_pgd_eps0.0500.bin"
# Record: pgd,0.05,device1_session1_pgd_eps0.0500.bin,<ber>,<per>,<total>,<good>,<bad>

echo "--- pgd eps=0.07 ---"
# python3 $WIFI_RX --input "$DIR/device1_session1_pgd_eps0.0700.bin"
# Record: pgd,0.07,device1_session1_pgd_eps0.0700.bin,<ber>,<per>,<total>,<good>,<bad>

echo "--- pgd eps=0.09 ---"
# python3 $WIFI_RX --input "$DIR/device1_session1_pgd_eps0.0900.bin"
# Record: pgd,0.09,device1_session1_pgd_eps0.0900.bin,<ber>,<per>,<total>,<good>,<bad>

echo "--- pgd eps=0.1 ---"
# python3 $WIFI_RX --input "$DIR/device1_session1_pgd_eps0.1000.bin"
# Record: pgd,0.1,device1_session1_pgd_eps0.1000.bin,<ber>,<per>,<total>,<good>,<bad>

