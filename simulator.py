# using GNPy v2.1

import getpass
import logging as lg
import time
import json
import networkx as nx

import gnpy.core.ansi_escapes as ansi_escapes
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from gnpy.core.elements import Transceiver, Fiber, RamanFiber, Edfa
from gnpy.core.equipment import (
    load_equipment,
    trx_mode_params,
)
from gnpy.core.exceptions import (
    ConfigurationError,
    EquipmentConfigError,
    NetworkTopologyError,
)
from gnpy.core.info import (
    SpectralInformation,
    Channel,
    Power,
    Pref,
)
from gnpy.core.network import build_network, SimParams, network_from_json
from gnpy.core.request import Path_request, compute_constrained_path
from gnpy.core.utils import db2lin, lin2db
from plotly.subplots import make_subplots
# from scripts.neteng.optical.bow.input_parameters import fiber_spans

logger = lg.getLogger(__name__)

# Simulation parameters
tx_awg_mean_loss = 6.0
tx_awg_rms_loss = 1.0 / 3.0
patch_cable_rms_loss = 1.0 / 3.0
tx_module_mean_output_power = -8.0
tx_module_rms_output_power = 2.0 / 3.0
zr_fbaud = 60e9
zr_roll_off = 0.2
zr_channel_spacing_ghz = 75.0
crosstalk_penalty_base_rsnr = 26.0 - lin2db(zr_fbaud / 12.5e9)
# ROSNR penalty due to crosstalk, given for 1 neighbor, as a function of aggressor to signal power in dB
crosstalk_penalty = {
    -10.0: 0.0,
    -1.0: 0.1,
    0.0: 0.2,
    1.0: 0.3,
    2.0: 0.5,
    3.0: 0.8,
    4.0: 1.2,
    5.0: 1.7,
    6.0: 2.3,
    7.0: 3.0,
}
transceiver_min_ch_spacing = zr_channel_spacing_ghz * 1e9


def format_gnpy_amp_network_element(
    end_port_name: str,
    amp_gain: float,
    type_variety: str,
    lat: float = 0.0,
    lon: float = 0.0,
):
    amp_element = {
        "uid": f"Line Amp:{end_port_name}",
        "type": "Edfa",
        "type_variety": type_variety,
        "operational": {
            "gain_target": amp_gain,
            "tilt_target": -0.5
            },
        "metadata": {
            "location": {
                "region": "",
                "latitude": lat,
                "longitude": lon,
            }
        },
    }
    return amp_element


def format_gnpy_fiber_network_element(
    uid: str,
    length_km: float,
    loss_db: float,
    type_variety: str = "SMF28",
    att_in: float = 1,
    con_in: float = 1,
    con_out: float = 1,
    latitude: float = 0.0,
    longitude: float = 0.0,
    region: str = "LH",
):
    loss_coef = (loss_db - att_in - con_in - con_out) / length_km
    fiber_element = {
        "uid": uid,
        "type": "Fiber",
        "type_variety": type_variety,
        "params": {
            "length": length_km,
            "loss_coef": loss_coef,
            "length_units": "km",
            "att_in": att_in,
            "con_in": con_in,
            "con_out": con_out,
        },
        "metadata": {
            "location": {
                "region": region,
                "latitude": latitude,
                "longitude": longitude,
            },
        },
    }
    return fiber_element


def format_gnpy_transponder_network_element(
    uid: str,
    type_variety: str = "vendorA_trx-type1",
    lat: float = 0.0,
    lon: float = 0.0,
    city: str = "",
    region: str = "",
):
    return {
        "uid": uid,
        "type": "Transceiver",
        "type_variety": type_variety,
        "metadata": {
            "city": city,
            "region": region,
            "latitutde": lat,
            "longitude": lon,
        },
    }


def Configure_GNPy_Network(amp_parameters, amp_type_id, fiber_spans):
    gnpy_network = {
        "network_name": "Southeast Asia Backbone",
        "elements": [],
        "connections": [],
    }
    src_transponder = format_gnpy_transponder_network_element("xpdr_start")
    gnpy_network["elements"].append(src_transponder)
    source = gnpy_network["elements"][0]["uid"]
    last_node = source

    for amp in amp_parameters:
        locals()["amp-" + amp] = format_gnpy_amp_network_element(
            end_port_name=amp,
            amp_gain=amp_parameters[amp],
            type_variety=amp_type_id[amp],
        )
        gnpy_network["elements"].append(locals()["amp-" + amp])
        connection = {"from_node": last_node, "to_node": locals()["amp-" + amp]["uid"]}
        gnpy_network["connections"].append(connection)
        last_node = locals()["amp-" + amp]["uid"]

        if amp in fiber_spans:
            fiber_element = format_gnpy_fiber_network_element(
                uid=fiber_spans[amp]["name"],
                length_km=fiber_spans[amp]["length"],
                loss_db=fiber_spans[amp]["loss"],
            )
            gnpy_network["elements"].append(fiber_element)
            connection = {"from_node": last_node, "to_node": fiber_element["uid"]}
            gnpy_network["connections"].append(connection)
            last_node = fiber_element["uid"]

    dst_transponder = format_gnpy_transponder_network_element("xpdr_end")
    gnpy_network["elements"].append(dst_transponder)
    dest = gnpy_network["elements"][-1]["uid"]
    connection = {"from_node": last_node, "to_node": dest}
    gnpy_network["connections"].append(connection)

    return gnpy_network, source, dest


def go_gnpy(
    path,
    equipment,
    channel_frequencies,
    crosstalk_penalty,
    sim_params,
    verbose,
    greenfield: bool = False,
    req_power=0.0,
    debug_requests="",
):
    nb_channel = len(channel_frequencies)
    pref_ch_db = lin2db(req_power * 1e3)

    spans = [
        s.length for s in path if isinstance(s, RamanFiber) or isinstance(s, Fiber)
    ]
    print(f"Propagating through {len(spans)} fiber spans over {sum(spans)/1000:.0f} km")

    try:
        p_start, p_stop, p_step = equipment["SI"]["default"].power_range_db

    except TypeError:
        print(
            "invalid power range definition in eqpt_config, should be power_range_db: [lower, upper, step]"
        )

    dp_db = 0.0  # Power offset, needs to include losses ahead of fiber span
    power_optimization_incomplete = True
    power_delta_range = [-5.0, 5.0]

    while power_optimization_incomplete:
        req_power = db2lin(pref_ch_db + dp_db) * 1e-3
        pin = lin2db(
            sum([db2lin(c["power"]) for c in channel_frequencies]) / nb_channel
        )
        print(
            f"Propagating with input power = {ansi_escapes.cyan}{lin2db(req_power*1e3):.2f} dBm{ansi_escapes.reset}:"
        )
        pref = Pref(lin2db(req_power * 1e3), pin, lin2db(nb_channel))
        propagated_carriers = []
        next_osnr_penalty = 0.0
        for ix in range(nb_channel):
            if ix < nb_channel - 1:
                power_difference = (
                    channel_frequencies[ix]["power"]
                    - channel_frequencies[ix + 1]["power"]
                )
                osnr_penalty = np.interp(
                    -power_difference,
                    list(crosstalk_penalty.keys()),
                    list(crosstalk_penalty.values()),
                )
                crosstalk_ase = (
                    db2lin(-crosstalk_penalty_base_rsnr)
                    - db2lin(-crosstalk_penalty_base_rsnr - next_osnr_penalty)
                    + db2lin(-crosstalk_penalty_base_rsnr)
                    - db2lin(-crosstalk_penalty_base_rsnr - osnr_penalty)
                )
                next_osnr_penalty = np.interp(
                    power_difference,
                    list(crosstalk_penalty.keys()),
                    list(crosstalk_penalty.values()),
                )
            else:
                crosstalk_ase = db2lin(-crosstalk_penalty_base_rsnr) - db2lin(
                    -crosstalk_penalty_base_rsnr - next_osnr_penalty
                )
            pch = db2lin(channel_frequencies[ix]["power"]) * 1e-3
            if verbose:
                print("pch", pch)
                print("crosstalk_ase", crosstalk_ase)
            new_ch = Channel(
                ix + 1,
                channel_frequencies[ix]["center_frequency"],
                channel_frequencies[ix]["fbaud"],
                channel_frequencies[ix]["roll_off"],
                Power(pch, 0, pch * crosstalk_ase),
            )
            propagated_carriers.append(new_ch)
        si = SpectralInformation(
            pref=pref,
            carriers=propagated_carriers,
        )

        infos = {}
        for el in path:
            before_si = si
            after_si = si = el(si)

            infos[el] = before_si, after_si
            print("el", el)

            if "path_snr_accumulation" in debug_requests:
                si_snr = []
                for si_carrier in si.carriers:
                    if si_carrier.power.ase:
                        si_snr.append(
                            si_carrier.power.signal
                            / (si_carrier.power.nli + si_carrier.power.ase)
                        )
                    else:
                        si_snr.append(None)

        if greenfield:  # Optimize launch power to balance ASE and NLI noise
            if dp_db <= power_delta_range[0] or dp_db >= power_delta_range[1]:
                power_optimization_incomplete = False
            else:
                ase_nli_power_imbal = (
                    np.mean(path[-1].osnr_ase) - np.mean(path[-1].osnr_nli) + 3.0
                )
                dp_db -= 0.33 * ase_nli_power_imbal
                dp_db = max([power_delta_range[0], dp_db])
                dp_db = min([power_delta_range[1], dp_db])
                power_optimization_incomplete = abs(ase_nli_power_imbal) > 0.05
        else:
            break

    if "show_devices" in debug_requests:
        print("=======")
        print("GNPy simulation results")
        for elem in path:
            print(elem)

    return path, infos, [lin2db(req_power * 1e3), dp_db]


def go_est_snr(
    amp_parameters,
    amp_type_id,
    fiber_spans,
    equipment,
    sim_params,
    channel_frequencies,
    crosstalk_penalty,
    debug_requests,
    verbose,
    end_node="a",
    greenfield=True,
):
    max_launch_power_a_per_50ghz_dbm = -2.0

    try:
        network_json, source_uid, destination_uid = Configure_GNPy_Network(
            amp_parameters, amp_type_id, fiber_spans
        )
    except NetworkTopologyError:
        return None, None, None, None
    if network_json is None:
        return None, None, None, None

    print("network_json", network_json)

    network = network_from_json(network_json, equipment)
    print("network", network)

    plt.figure(figsize=(8,8))
    nx.draw(network, with_labels=True, node_color="white", edgecolors="blue", node_size=500)
    plt.show() 

    min_ch_spacing = transceiver_min_ch_spacing
    launch_power_dbm = max_launch_power_a_per_50ghz_dbm + lin2db(min_ch_spacing / 50e9)
    transceivers = {n.uid: n for n in network.nodes() if isinstance(n, Transceiver)}
    source = transceivers[source_uid]
    destination = transceivers[destination_uid]

    if not transceivers:
        exit("Network has no transceivers!")
    if len(transceivers) < 2:
        exit("Network has only one transceiver!")

    # If no partial match or no source/destination provided pick random
    if not source:
        source = list(transceivers.values())[0]
        del transceivers[source.uid]

    if not destination:
        destination = list(transceivers.values())[0]

    params = {}
    params["request_id"] = 0
    params["trx_type"] = ""
    params["trx_mode"] = ""
    params["source"] = source.uid
    params["destination"] = destination.uid
    params["nodes_list"] = [destination.uid]
    params["loose_list"] = ["strict"]
    params["format"] = ""
    params["path_bandwidth"] = 0
    params["bidir"] = False
    trx_params = trx_mode_params(equipment)
    if launch_power_dbm:
        trx_params["power"] = db2lin(float(launch_power_dbm)) * 1e-3
    params.update(trx_params)
    req = Path_request(**params)
    nb_channel = len(channel_frequencies)
    pref_ch_db = lin2db(req.power * 1e3)  # reference channel power / span (SL=20dB)
    pref_total_db = pref_ch_db + lin2db(nb_channel)
    build_network(network, equipment, pref_ch_db, pref_total_db)
    path = compute_constrained_path(network, req)
    print("path: ", path)

    if verbose:
        print("=======")
        print("before GNPy simulation")
        for elem in path:
            print(elem)

    path, infos, launch_power = go_gnpy(
        path,
        equipment,
        channel_frequencies,
        crosstalk_penalty,
        sim_params,
        verbose,
        greenfield=greenfield,
        req_power=req.power,
        debug_requests=debug_requests,
    )
    # save_network(filename, network)

    if "show_channels" in debug_requests:
        print("\nThe total SNR per channel (in signal BW) at the end of the line is:")
        print(
            "{:>5}{:>15}{:>14}{:>12}{:>12}{:>12}".format(
                "Ch. #",
                "Ch freq (THz)",
                "Ch Pwr (dBm)",
                "OSNR ASE",
                "SNR NLI",
                "SNR total",
            )
        )
        for final_carrier, ch_osnr, ch_snr_nl, ch_snr in zip(
            infos[path[-1]][1].carriers,
            path[-1].osnr_ase,
            path[-1].osnr_nli,
            path[-1].snr,
        ):
            ch_freq = final_carrier.frequency * 1e-12
            ch_power = lin2db(final_carrier.power.signal * 1e3)
            print(
                "{:5}{:15.4f}{:14.2f}{:12.2f}{:12.2f}{:12.2f}".format(
                    final_carrier.channel_number,
                    round(ch_freq, 4),
                    round(ch_power, 2),
                    round(ch_osnr, 2),
                    round(ch_snr_nl, 2),
                    round(ch_snr, 2),
                )
            )

    if "do_plot" in debug_requests:
        fig = plt.figure(figsize=(14, 6))
        ax = fig.add_subplot(111)
        xpl = [0.0]
        ypl = [0.0]
        dd = 0.0
        ttx = []
        tty = []
        tt = []
        for ix in range(len(path)):
            if type(path[ix]) is Transceiver:
                ttx.append(xpl[-1])
                tty.append(ypl[-1])
                tt.append(path[ix].uid)
            if type(path[ix]) is Edfa:
                xpl.append(xpl[-1] + dd)
                ypl.append(path[ix].pin_db)
                xpl.append(xpl[-1])
                ypl.append(path[ix].pout_db)
                ttx.append(xpl[-1])
                tty.append(ypl[-1])
                tt1 = path[ix].uid.split(":")
                tt.append(tt1[1])
                dd = 0.0
            if type(path[ix]) is Fiber:
                dd += path[ix].length * 1e-3
        ax.plot(xpl, ypl)
        for ix in range(len(tt)):
            ax.text(ttx[ix], tty[ix], tt[ix], rotation=90)
        ax.set(xlabel="Route distance [km]", ylabel="Aggregate Power [dBm]")
    return path, infos, launch_power


def run_simulator(
    amp_parameters,
    amp_type_id,
    fiber_spans,
    left_freq_mhz,
    right_freq_mhz,
    verbose,
):
    USER = getpass.getuser()
    channel_osnr = {}
    channel_power = {}
    file_path = f"./data/"
    equipment_file = file_path + "eqpt_config.json"

    sim_params = SimParams(
        {
            "raman_computed_channels": [1, 18, 37, 56, 75],
            "raman_parameters": {
                "flag_raman": True,
                "space_resolution": 10e3,
                "tolerance": 1e-8,
            },
            "nli_parameters": {
                "nli_method_name": "ggn_spectrally_separated",
                "wdm_grid_size": zr_channel_spacing_ghz * 1e9,
                "dispersion_tolerance": 1,
                "phase_shift_tollerance": 0.1,
            },
        }
    )

    try:
        equipment = load_equipment(equipment_file)
    except EquipmentConfigError as e:
        print(
            f"{ansi_escapes.red}Configuration error in the equipment library:{ansi_escapes.reset} {e}"
        )
        exit(1)
    except NetworkTopologyError as e:
        print(f"{ansi_escapes.red}Invalid network definition:{ansi_escapes.reset} {e}")
        exit(1)
    except ConfigurationError as e:
        print(f"{ansi_escapes.red}Configuration error:{ansi_escapes.reset} {e}")
        exit(1)

    channel_frequencies = []
    ## uniform spectrum
    # channel_num = int((right_freq_mhz - left_freq_mhz) / 75000) - 1  ## channel spacing is 75 GHz
    # chf = np.linspace(
    #     left_freq_mhz + zr_channel_spacing_ghz * 1e3 / 2,
    #     right_freq_mhz - zr_channel_spacing_ghz * 1e3 / 2,
    #     channel_num,
    # )
    ## other spectrum
    channel_num = 8
    chf = [
        193087500,  # CH8
        193237500,  # CH7
        193387500,  # CH6
        193687500,  # CH5
        193837500,  # CH4
        193987500,  # CH3
        194137500,  # CH2
        194287500,  # CH1
    ]

    for ch_ix in range(len(chf)):
        tx_pwr = (
            tx_module_mean_output_power
            - tx_awg_mean_loss
            + float(np.random.randn(1)) * tx_awg_rms_loss
            - abs(float(np.random.randn(1)) * patch_cable_rms_loss)
            + float(np.random.randn(1)) * tx_module_rms_output_power
        )
        channel_frequencies.append(
            {
                "center_frequency": chf[ch_ix],
                "fbaud": zr_fbaud,
                "roll_off": zr_roll_off,
                "width": zr_channel_spacing_ghz * 1e9,
                "power": tx_pwr,
            }
        )
    t0 = time.perf_counter()
    debug_requests = [
        "do_plot",
        "show_channels",
        "show_devices",
    ]
    path, infos, launch_power = go_est_snr(
        amp_parameters,
        amp_type_id,
        fiber_spans,
        equipment,
        sim_params,
        channel_frequencies,
        crosstalk_penalty,
        debug_requests=debug_requests,
        verbose=verbose,
        end_node="a",
        greenfield=False,
    )
    t1 = time.perf_counter()
    print(f"Elapsed time for GNPy calc is {t1-t0:.2f} seconds")

    snr_required = 27 - lin2db(zr_fbaud / 12.5e9)
    if verbose:
        print("\nThe total SNR per channel (in signal BW) at the end of the line is:")
        print(
            "{:>5}{:>15}{:>14}{:>12}{:>12}{:>12}".format(
                "Ch. #",
                "Ch freq (MHz)",
                "Ch Pwr (dBm)",
                "OSNR ASE",
                "SNR NLI",
                "SNR total",
            )
        )

    for final_carrier, ch_osnr, ch_snr_nl, ch_snr in zip(
        infos[path[-1]][1].carriers, path[-1].osnr_ase, path[-1].osnr_nli, path[-1].snr
    ):
        ch_freq = int(final_carrier.frequency)  # MHz
        ch_power = lin2db(final_carrier.power.signal * 1e3)
        channel_osnr[ch_freq] = round(ch_snr, 2)
        channel_power[ch_freq] = round(ch_power, 2)
        if verbose:
            if ch_osnr < snr_required:
                print(
                    f"{final_carrier.channel_number:5}{ch_freq:15.4f}{ch_power:14.2f}{ch_osnr:12.2f}"
                    f"{ch_snr_nl:12.2f}{ansi_escapes.red}{ch_snr:12.2f}{ansi_escapes.reset}"
                )
            else:
                print(
                    "{:5}{:15.4f}{:14.2f}{:12.2f}{:12.2f}{:12.2f}".format(
                        final_carrier.channel_number,
                        ch_freq,
                        round(ch_power, 2),
                        round(ch_osnr, 2),
                        round(ch_snr_nl, 2),
                        round(ch_snr, 2),
                    )
                )

    ch_freq = []
    ch_power = [c["power"] for c in channel_frequencies]
    snr = []
    for final_carrier, _ch_osnr, _ch_snr_nl, ch_snr in zip(
        infos[path[-1]][1].carriers, path[-1].osnr_ase, path[-1].osnr_nli, path[-1].snr
    ):
        ch_freq.append(final_carrier.frequency * 1e-12)
        snr.append(ch_snr)

    return channel_osnr, channel_power


if __name__ == "__main__":
    ## example data to test the GNPy simulator
    parameters = {
        "amp1": 14.2,
        "amp2": 16.6,
        "amp3": 18.8,
        "amp4": 18.2, 
        "amp5": 16.8, 
        "amp6": 15.8, 
        "amp7": 15.0, 
        "amp8": 14.5,
    }
    amp_type_id = {
        "amp1": "high_detail_model_example",
        "amp2": "high_detail_model_example",
        "amp3": "high_detail_model_example",
        "amp4": "high_detail_model_example", 
        "amp5": "high_detail_model_example", 
        "amp6": "high_detail_model_example", 
        "amp7": "high_detail_model_example", 
        "amp8": "high_detail_model_example",
    }
    fiber_spans = {
        "amp1": {
            "name": "amp1=amp2",
            "length": 42,
            "loss": 16.5,
        },
        "amp2": {
            "name": "amp2=amp3",
            "length": 45,
            "loss": 14.25,
        },
        "amp3": {
            "name": "amp3=amp4",
            "length": 74,
            "loss": 21.5,
        },
        "amp4": {
            "name": "amp4=amp5",
            "length": 66,
            "loss": 19.5,
        },
        "amp5": {
            "name": "amp5=amp6",
            "length": 65,
            "loss": 19.25,
        },
        "amp6": {
            "name": "amp6=amp7",
            "length": 64, 
            "loss": 19
        },
        "amp7": {
            "name": "amp7=amp8",
            "length": 32,
            "loss": 14
        },
    }
    left_freq = 193050000
    right_freq = 194325000

    estimated_channel_osnr = {}
    for _ in range(10):
        estimated_channel_osnr_current, estimated_channel_power_current = run_simulator(
            parameters,
            amp_type_id,
            fiber_spans,
            left_freq - 37500,
            right_freq + 37500,
            verbose=True,
        )  # 4800 GHz C-band spectrum, @75 GHz spacing
        if len(estimated_channel_osnr) > 0:
            for ch in estimated_channel_osnr_current:
                estimated_channel_osnr[ch] += estimated_channel_osnr_current[ch]
        else:
            for ch in estimated_channel_osnr_current:
                estimated_channel_osnr[ch] = estimated_channel_osnr_current[ch]
        print(estimated_channel_osnr_current)
    for ch in estimated_channel_osnr:
        estimated_channel_osnr[ch] = estimated_channel_osnr[ch]/10

    print("estimated_channel_osnr", estimated_channel_osnr)