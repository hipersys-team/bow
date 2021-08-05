import concurrent.futures
import sys
import time

from bayesian_amp_control import get_line_osnr
from fcr_interface import (
    Line_Device,
    get_transponder_pm,
    get_transponder_adj,
)
from input_parameters import (
    amplifiers_location,
    wss_location,
    proprietary_controller_info,
    transponder_box,
    wss_onoff,
)


# polling the performance of amplifiers
def monitor_amp(iter_num, line):
    timestamp = time.localtime()
    pt_time = time.strftime("%Y.%m.%d.%H.%M.%S", timestamp)

    gain_time_series = []
    parameter_history = []

    t = 0
    while t < iter_num:  # repetitive measurement
        gain_results = line.read_amp(line.amplifiers)
        gain_results_float = []
        gain_results_key = []
        for x in gain_results:
            gain_results_float.append(float(gain_results[x]))
            gain_results_key.append(x)
        gain_results_zip = dict(zip(gain_results_key, gain_results_float))
        parameter_history.append(gain_results_zip)

        timestamp = time.localtime()
        pt_time = time.strftime("%Y.%m.%d.%H.%M.%S", timestamp)
        gain_time_series.append(pt_time)

        print("\033[1;31m this is the " + str(t) + " amp query\033[0;0m")
        print("amp gain_results_zip", gain_results_zip)
        print("amp pt_time", pt_time)
        t = t + 1

    return parameter_history, gain_time_series


# polling the performance of transponders
def monitor_transponder(iter_num, line, ssh_list, metric_bit):
    t = 0
    ber_history = []
    q_history = []
    esnr_history = []
    bermax_history = []
    qmin_history = []
    esnrmin_history = []
    tx_pm_time_series = []

    while t < iter_num:  # repetitive measurement
        (
            transponder_Q_set,
            transponder_QMin_set,
            transponder_ber_set,
            transponder_berMax_set,
            transponder_esnr_set,
            transponder_esnrMin_set,
        ) = get_transponder_pm(
            line,
            ssh_list,
            metric_bit,
            ssh_flag=0,
        )
        timestamp = time.localtime()
        pt_time = time.strftime("%Y.%m.%d.%H.%M.%S", timestamp)
        tx_pm_time_series.append(pt_time)

        print("\033[1;34m this is the " + str(t) + " transponder query\033[0;0m")
        print("current Q", transponder_Q_set)
        print("current BER", transponder_ber_set)
        print("current ESNR", transponder_esnr_set)
        print("current Q Min", transponder_QMin_set)
        print("current BER Max", transponder_berMax_set)
        print("current ESNR Min", transponder_esnrMin_set)
        print("transponder pt_time", pt_time)

        ber_history.append(transponder_ber_set)
        q_history.append(transponder_Q_set)
        esnr_history.append(transponder_esnr_set)
        bermax_history.append(transponder_berMax_set)
        qmin_history.append(transponder_QMin_set)
        esnrmin_history.append(transponder_esnrMin_set)
        t = t + 1

    return (
        ber_history,
        q_history,
        esnr_history,
        bermax_history,
        qmin_history,
        esnrmin_history,
        tx_pm_time_series,
    )


# polling the per-channel osnr
def monitor_wss(iter_num, line, ssh_list, ssh_flag):
    t = 0
    osnr_history = []
    estimated_channel_osnr = []
    estimated_channel_power = []
    plot_flag = 0
    time_series = []

    while t < iter_num:  # repetitive measurement
        current_osnr = get_line_osnr(
            line,
            ssh_list,
            estimated_channel_osnr,
            estimated_channel_power,
            plot_flag,
            ssh_flag,
        )
        osnr_history.append(current_osnr)
        timestamp = time.localtime()
        pt_time = time.strftime("%Y.%m.%d.%H.%M.%S", timestamp)
        time_series.append(pt_time)
        print("\033[1;32m this is the " + str(t) + " WSS query\033[0;0m")
        print(current_osnr)
        print("OSNR pt_time", pt_time)
        t = t + 1

    return osnr_history, time_series


if __name__ == "__main__":
    transponder_mod, transponder_fre = get_transponder_adj(
        transponder_box, metric_bit=[1, 1]
    )
    print("transponder_mod", transponder_mod)
    print("transponder_fre", transponder_fre)

    # define line devices
    line = Line_Device(
        amplifiers_location,
        wss_location,
        wss_onoff,
        proprietary_controller_info,
        transponder_box,
        transponder_fre,
        fast_slow_version="fast",
    )  

    iter_num = 30
    if line.fast_slow_version == "slow":
        metric_bit = [1, 1, 1, 1, 1, 1]
    else:
        metric_bit = [1, 0, 1, 0, 1, 0]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        amp = executor.submit(
            monitor_amp,
            iter_num,
            line,
        )
        transponder = executor.submit(
            monitor_transponder,
            iter_num,
            line,
            metric_bit,
        )

        parameter_history, gain_time_series = amp.result()
        ber_history, q_history, esnr_history, tx_pm_time_series = transponder.result()

    sys.exit(0)
