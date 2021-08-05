import time
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np


def plot_channel(
    channel_history,
    colors,
    pt_time,
    sobol_index,
    best_index,
    plots_location,
    metric_name,
):
    wave_simple = []
    for w in channel_history[1]:
        locals()[metric_name + str(w)] = []
        wave_simple.append(w)
    color_dict = dict(zip(wave_simple, colors))

    for t in range(len(channel_history)):
        for q in channel_history[t]:
            locals()[metric_name + str(q)].append(channel_history[t][q])
    x = range(len(channel_history))
    plt.figure(figsize=(10, 3))
    for w in channel_history[0]:
        plt.scatter(
            x[0], locals()[metric_name + str(w)][0], marker="o", color=color_dict[w]
        )
        plt.plot(
            x[1:-1],
            locals()[metric_name + str(w)][1:-1],
            label="channel_" + str(w),
            linewidth=1,
            marker="o",
            color=color_dict[w],
        )
        plt.scatter(
            x[-1],
            locals()[metric_name + str(w)][-1],
            marker="*",
            color=color_dict[w],
        )
    plt.legend(loc="lower right")
    plt.xlabel("BO trials")
    plt.ylabel(metric_name)
    if metric_name == "BER":
        plt.yscale("log")
    plt.xticks(np.arange(30))
    plt.axvline(x=sobol_index, color="gray", linestyle="--")
    plt.axvline(x=best_index, color="gray", linestyle="--")
    plt.tight_layout()
    figure_name = metric_name + "_history_timestamp.png"
    figure_name = figure_name.replace("timestamp", pt_time)
    figure_name = plots_location + figure_name
    plt.savefig(figure_name, dpi=100, bbox_inches="tight")
    plt.show()



def plot_spectrum(
    line,
    initial_spectrum,
    initial_noise,
    central_freq,
    spectrum_detail,
    freq_detail,
    bins,
    estimated_channel_power,
):
    float_freq_detail = [float(x.split("-")[-1]) / 1000000 for x in freq_detail]
    float_central_freq = [x / 1000000 for x in central_freq]
    print("float_central_freq", float_central_freq)
    print("estimated_channel_power", estimated_channel_power)
    estimate_channel_power_list = []
    if len(estimated_channel_power) > 0:
        for ch_thz in float_central_freq:
            for est_ch_mhz in estimated_channel_power:
                if abs(ch_thz * 1000000 - est_ch_mhz) < 75000:
                    estimate_channel_power_list.append(
                        estimated_channel_power[est_ch_mhz]
                    )
                    break
    location = (
        line.wss["roadm_id"]
        + ":"
        + str(line.wss["chassis_id"])
        + "-"
        + str(line.wss["card_id"])
        + "-"
        + str(line.wss["port_id"])
    )
    plt.figure(figsize=(15, 3))
    plt.plot(
        float_freq_detail,
        spectrum_detail,
        linewidth=1,
        label=location,
        color="blue",
    )
    for b in bins:
        plt.axvline(x=float(b[0]) / 1000000, color="gray", linestyle="--", linewidth=1)
        plt.axvline(x=float(b[-1]) / 1000000, color="red", linestyle="--", linewidth=1)
    if len(float_central_freq) == len(initial_spectrum):
        plt.scatter(float_central_freq, initial_spectrum, marker="*", color="blue")
    if len(float_central_freq) == len(initial_noise):
        plt.scatter(float_central_freq, initial_noise, marker="*", color="blue")
    if len(estimated_channel_power) > 0:
        plt.scatter(
            float_central_freq,
            estimate_channel_power_list,
            marker="o",
            color="green",
            label="GNPy estimation",
        )
    plt.ylabel("Power (dBm)")
    plt.xlabel("Channel frequency (THz)")
    plt.ylim(-30, 0)
    plt.legend(loc="best", fontsize=14)
    plt.show()


def line_plot(
    plots_location,
    amplifiers_location,
    best_parameters,
    best_index,
    sobol_index,
    parameter_history,
    reward_history,
    osnr_history,
    ber_history,
    q_history,
    esnr_history,
):
    x = range(len(parameter_history))
    timestamp = time.localtime()
    pt_time = time.strftime("%Y.%m.%d.%H.%M.%S", timestamp)
    colors = [
        "blue",
        "orange",
        "green",
        "red",
        "purple",
        "brown",
        "black",
        "pink",
    ]  # color code for 8 items

    ## plot gain history
    plt.figure(figsize=(10, 3))
    amps_simple = []
    sum_gain = []
    for a in amplifiers_location:
        locals()["gain" + a] = []
        amps_simple.append(a)
    for param in parameter_history:
        current_gain = 0
        for a in param:
            locals()["gain" + a].append(float(param[a]))
            current_gain += float(param[a])
        sum_gain.append(current_gain)

    color_dict = dict(zip(amps_simple, colors))
    for a in amps_simple:
        if len(x) == len(locals()["gain" + a]):
            plt.scatter(x[0], locals()["gain" + a][0], color=color_dict[a], marker="o")
            plt.plot(
                x[1:],
                locals()["gain" + a][1:],
                label=a,
                color=color_dict[a],
                linewidth=1,
                marker="o",
            )
            plt.scatter(
                x[best_index],
                locals()["gain" + a][best_index],
                color=color_dict[a],
                marker="*",
            )
        else:
            print("plot vector size not equal")
            print("x", x)
            print(a, locals()["gain" + a])
    plt.legend(loc="lower right", bbox_to_anchor=(1, 0.5))
    plt.xlabel("BO trials")
    plt.ylabel("amplifier parameter value")
    plt.xticks(np.arange(30))
    plt.axvline(x=sobol_index, color="gray", linestyle="--")
    plt.axvline(x=best_index, color="gray", linestyle="--")
    plt.tight_layout()
    figure_name = "param_history_timestamp.png"
    figure_name = figure_name.replace("timestamp", pt_time)
    figure_name = plots_location + figure_name
    plt.savefig(figure_name, dpi=100, bbox_inches="tight")

    ## plot sum gain history
    plt.figure(figsize=(10, 3))
    if len(x) == len(sum_gain):
        plt.scatter(x[0], sum_gain[0], marker="o")
        plt.plot(
            x[1:],
            sum_gain[1:],
            linewidth=1,
            marker="o",
        )
    else:
        print("plot vector size not equal")
        print("x", x)
        print("sum_gain", sum_gain)
    plt.xlabel("BO trials")
    plt.ylabel("SUM amplifier parameter value")
    plt.xticks(np.arange(30))
    plt.axvline(x=sobol_index, color="gray", linestyle="--")
    plt.axvline(x=best_index, color="gray", linestyle="--")
    plt.tight_layout()
    figure_name = "sumgain_history_timestamp.png"
    figure_name = figure_name.replace("timestamp", pt_time)
    figure_name = plots_location + figure_name
    plt.savefig(figure_name, dpi=100, bbox_inches="tight")
    plt.show()

    ## plot reward value history
    plt.figure(figsize=(10, 3))
    rewards = []
    for xs in reward_history:
        rewards.append(xs["bow"][0])
    x = range(len(rewards))
    if len(x) == len(rewards):
        plt.scatter(x[0], rewards[0], marker="o")
        plt.plot(
            x[1:-1],
            rewards[1:-1],
            linewidth=1,
            marker="o",
        )
        plt.scatter(x[-1] + 1, rewards[-1], marker="*")
    else:
        print("plot vector size not equal")
        print("x", x)
        print("rewards", rewards)
    plt.xlabel("BO trials")
    plt.ylabel("reward")
    plt.xticks(np.arange(30))
    plt.axvline(x=sobol_index, color="gray", linestyle="--")
    plt.axvline(x=best_index, color="gray", linestyle="--")
    plt.tight_layout()
    figure_name = "reward_history_timestamp.png"
    figure_name = figure_name.replace("timestamp", pt_time)
    figure_name = plots_location + figure_name
    plt.savefig(figure_name, dpi=100, bbox_inches="tight")
    plt.show()

    # plot osnr of each wavelengths history
    plot_channel(
        osnr_history,
        colors,
        pt_time,
        sobol_index,
        best_index,
        plots_location,
        metric_name="osnr",
    )

    # plot BER of each wavelengths history
    plot_channel(
        ber_history,
        colors,
        pt_time,
        sobol_index,
        best_index,
        plots_location,
        metric_name="BER",
    )

    # plot Q of each wavelengths history
    plot_channel(
        q_history,
        colors,
        pt_time,
        sobol_index,
        best_index,
        plots_location,
        metric_name="Q",
    )

    # plot ESNR of each wavelengths history
    plot_channel(
        esnr_history,
        colors,
        pt_time,
        sobol_index,
        best_index,
        plots_location,
        metric_name="esnr",
    )


def bar_plot(
    plots_location,
    write_delay_history,
    read_osnr_delay_history,
    read_transponder_delay_history,
    bo_acquisition_delay_history,
    total_delay_history,
):
    timestamp = time.localtime()
    pt_time = time.strftime("%Y.%m.%d.%H.%M.%S", timestamp)
    x = range(len(write_delay_history))
    plt.figure(figsize=(10, 3))
    plt.bar(x, write_delay_history, label="write_delay")
    plt.bar(x, read_osnr_delay_history, bottom=write_delay_history, label="osnr_delay")
    plt.bar(
        x,
        read_transponder_delay_history,
        bottom=[
            write_delay_history[t] + read_osnr_delay_history[t]
            for t in range(len(read_osnr_delay_history))
        ],
        label="transponder_delay",
    )
    plt.bar(
        x,
        bo_acquisition_delay_history,
        bottom=[
            write_delay_history[t]
            + read_osnr_delay_history[t]
            + read_transponder_delay_history[t]
            for t in range(len(read_osnr_delay_history))
        ],
        label="BO acquisition delay",
    )
    plt.xlabel("BO trials")
    plt.ylabel("Operation delay (s)")
    plt.xticks(np.arange(30))
    plt.legend(loc="lower right", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    figure_name = "decompose_delay_timestamp.png"
    figure_name = figure_name.replace("timestamp", pt_time)
    figure_name = plots_location + figure_name
    plt.savefig(figure_name, dpi=100, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(10, 3))
    plt.bar(x, total_delay_history, label="total delay")
    plt.xlabel("BO trials")
    plt.ylabel("Operation delay (s)")
    plt.xticks(np.arange(30))
    plt.legend(loc="lower right", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    figure_name = "total_delay_timestamp.png"
    figure_name = figure_name.replace("timestamp", pt_time)
    figure_name = plots_location + figure_name
    plt.savefig(figure_name, dpi=100, bbox_inches="tight")
    plt.show()


def violin_plot(
    plots_location,
    parameter_all,
    param_range,
    pt_time,
    xlabelname,
    ylabelname,
    figure_name,
    log_y,
):
    violin_list = []
    amp_set = []
    sorted_parameter_0 = OrderedDict(
        sorted(parameter_all[0].items(), key=lambda t: t[0])
    )
    for n in sorted_parameter_0:
        locals()["violin-" + str(n)] = []
        amp_set.append(n)

    for i in range(len(parameter_all)):
        sorted_parameter = OrderedDict(
            sorted(parameter_all[i].items(), key=lambda t: t[0])
        )
        for j in sorted_parameter:
            locals()["violin-" + str(j)].append(sorted_parameter[j])

    for n in sorted_parameter_0:
        violin_list.append(locals()["violin-" + str(n)])

    plt.figure()
    plt.violinplot(violin_list, showmeans=True, showmedians=False)
    dev_index = 0
    for dev in amp_set:
        if dev in param_range:
            dev_index += 1
            for bounds in param_range[dev]:
                plt.scatter(dev_index, bounds, color="black", marker="o")
    plt.xticks([y + 1 for y in range(len(violin_list))], amp_set)
    plt.xticks(rotation=90)  # Rotates X-Axis Ticks by 45-degrees
    plt.xlabel(xlabelname)
    plt.ylabel(ylabelname)
    if log_y:
        plt.yscale("log")
    plt.tight_layout()
    figure_name = figure_name.replace("timestamp", pt_time)
    figure_name = plots_location + figure_name
    plt.savefig(figure_name, dpi=100, bbox_inches="tight")
    plt.show()
