import concurrent.futures
import json
import math
import random
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import pyjq
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient
from crypto.keychain_service.keychain import ttypes as keychain
from fbnet.command_runner.CommandRunner.Command import Client as FcrClient
from fbnet.command_runner.CommandRunner.ttypes import Device
from fbnet.command_runner.CommandRunner.ttypes import SessionType, SessionData
from libfb.py.thrift import get_sr_client
from libfb.py.thrift_clients.keychain_thrift_client import KeychainClient
from gnpy.core.utils import db2lin, lin2db

from fcr_interface import clean_transponder
from input_parameters import amp_type_id
from plotting import plot_spectrum
from simulator import run_simulator
from ssh_connection import (
    ssh_setup,
    ssh_close,
)


FCR_PROD_TIER = "netsystems.fbnet_command_runner"


def get_secret(name, group):
    req = keychain.GetSecretRequest(
        name=name,
        group=group,
    )
    try:
        return KeychainClient().getSecret(req)
    except keychain.KeychainServiceException as ex:
        print("Error retrieving secret:" + ex)
        return False


def run_command(device_commands, read_or_write):
    try:
        user = get_secret("TACACS_USERNAME", "NETENG_AUTOMATION").secret
        pw = get_secret("TACACS_PASSWORD", "NETENG_AUTOMATION").secret
        if read_or_write == 1:
            persistent_sign = "auto"
        else:
            persistent_sign = "auto"
        fcr_commands = {
            Device(
                hostname=device,
                username=user,
                password=pw,
                session_type=SessionType.TL1,
                session_data=SessionData(
                    extra_options={
                        "format": "json",
                        "use_persistent_connection": persistent_sign,
                    }
                ),
            ): device_commands[device]
            for device in device_commands
        }
        with get_sr_client(
            FcrClient,
            tier=FCR_PROD_TIER,
        ) as client:
            res = client.bulk_run(fcr_commands)
            timestamp = time.localtime()
            timestamp = time.strftime("%Y.%m.%d.%H.%M.%S", timestamp)
        return res, timestamp

    except Exception as ex:
        print("User exception: {}".format(ex))
        timestamp = time.localtime()
        timestamp = time.strftime("%Y.%m.%d.%H.%M.%S", timestamp)
        return 1, timestamp


def read_proprietary_controller(proprietary_controller_node, proprietary_controller_id):
    read_proprietary_controllerler_cmd = "XXXXXXXXXXX"  ## anonymized southbound command

    commands = defaultdict(list)
    commands[proprietary_controller_node] = [read_proprietary_controller_cmd]

    results = 1
    count = 0
    while results == 1 and count < 10:  # rc = 1 means exception
        results, timestamp = run_command(commands, read_or_write=1)
        count += 1

    if results == 1:
        print("the FCR fails to fetch data")
        stat = "N/A"
        return stat, timestamp
    else:
        print(results[proprietary_controller_node])
        vars = {}
        if results[proprietary_controller_node][0].output == "":
            stat = "N/A"
        else:
            j_data = json.loads(results[proprietary_controller_node][0].output)
            jqscript1 = ".[][].fields[3]"
            primary = pyjq.all(jqscript1, j_data, vars=vars)
            print("proprietary_controller primary state:", primary[0])
            stat = primary[0]

        return stat, timestamp


def control_proprietary_controller(proprietary_controller_node, proprietary_controller_id, on_off_flag):
    enable_proprietary_controller_cmd = "XXXXXXXXXXX"  ## anonymized southbound command
    dis_proprietary_controller_cmd = "XXXXXXXXXXX"  ## anonymized southbound command

    commands = defaultdict(list)

    if on_off_flag == 1:
        commands[proprietary_controller_node] = [enable_proprietary_controller_cmd]
        print(commands)
        res, timestamp = run_command(commands, read_or_write=0)

    else:
        commands[proprietary_controller_node] = [dis_proprietary_controller_cmd]
        print(commands)
        res, timestamp = run_command(commands, read_or_write=0)

    return res, timestamp


def optimize_proprietary_controller(proprietary_controller_node, proprietary_controller_id):
    reopt_proprietary_controller_cmd = "XXXXXXXXXXX"  ## anonymized southbound command

    commands = defaultdict(list)
    commands[proprietary_controller_node] = [reopt_proprietary_controller_cmd]
    print(commands)
    res, timestamp = run_command(commands, read_or_write=0)

    return res, timestamp


def WSS_ONOFF_bulkctl(node_id, ONOFF_names, opqstatus):
    ctl_WSS_ONOFF = "XXXXXXXXXXX"  ## anonymized southbound command
    current_cmds = []
    for current_name in ONOFF_names:
        cmd = ctl_WSS_ONOFF.replace("ONOFF_ID", current_name)
        cmd = cmd.replace("OPQ_STATUS", opqstatus[ONOFF_names.index(current_name)])
        current_cmds.append(cmd)

    commands = defaultdict(list)
    commands[node_id] = current_cmds

    rc, _ = run_command(commands, read_or_write=0)

    return rc


def proprietary_controller_prepareness(proprietary_controller_node, proprietary_controller_id, final_state, verbose):
    readings, timestamp = read_proprietary_controller(proprietary_controller_node, proprietary_controller_id)
    if verbose:
        print("read_proprietary_controller command 1", readings, timestamp)
    # the final target is to disable proprietary_controller for BO amp configuration
    if final_state == "OUT_OF_SERVICE":
        if readings == "OUT_OF_SERVICE":
            res, timestamp = control_proprietary_controller(proprietary_controller_node, proprietary_controller_id, 1)
            if verbose:
                print("control_proprietary_controller command 2", res, timestamp)
            time.sleep(20)
            res, timestamp = optimize_proprietary_controller(proprietary_controller_node, proprietary_controller_id)
            if verbose:
                print("optimize_proprietary_controller command 3", res, timestamp)
            time.sleep(20)
            res, timestamp = control_proprietary_controller(proprietary_controller_node, proprietary_controller_id, 0)
            if verbose:
                print("control_proprietary_controller command 4", res, timestamp)
            time.sleep(20)
        else:
            res, timestamp = optimize_proprietary_controller(proprietary_controller_node, proprietary_controller_id)
            if verbose:
                print("optimize_proprietary_controller command 2", res, timestamp)
            time.sleep(20)
            res, timestamp = control_proprietary_controller(proprietary_controller_node, proprietary_controller_id, 0)
            if verbose:
                print("control_proprietary_controller command 3", res, timestamp)
            time.sleep(20)
    # the final target is to enable proprietary_controller for channel on off state config
    else:
        if readings == "OUT_OF_SERVICE":
            res, timestamp = control_proprietary_controller(proprietary_controller_node, proprietary_controller_id, 1)
            if verbose:
                print("control_proprietary_controller command 2", res, timestamp)
            time.sleep(20)
            res, timestamp = optimize_proprietary_controller(proprietary_controller_node, proprietary_controller_id)
            if verbose:
                print("optimize_proprietary_controller command 3", res, timestamp)
            time.sleep(20)
        else:
            res, timestamp = optimize_proprietary_controller(proprietary_controller_node, proprietary_controller_id)
            if verbose:
                print("optimize_proprietary_controller command 2", res, timestamp)
            time.sleep(20)
    return


def get_line_osnr(
    line, ssh_list, estimated_channel_osnr, estimated_channel_power, plot_flag, ssh_flag
):
    # read the performance metrics on related devices after amplifier change
    (
        new_spectrum,
        new_noise,
        central_freq,
        fine_grain_spectrum,
        fine_grain_frequency,
        bins,
    ) = line.read_spectrum(line.wss, ssh_list, ssh_flag)

    if plot_flag:
        plot_spectrum(
            line,
            new_spectrum,
            new_noise,
            central_freq,
            fine_grain_spectrum,
            fine_grain_frequency,
            bins,
            estimated_channel_power,
        )

    current_osnr = {}
    for tx in line.transponder_fre:
        current_osnr[line.transponder_fre[tx]] = None  # initiate osnr dict

    for ch in range(len(new_spectrum)):
        if new_spectrum[ch] > -15:  # regular channel power should be larger than -15
            if (
                new_spectrum[ch] - new_noise[ch] > 8
            ):  # channel should have larger than 8 db osnr
                current_osnr[central_freq[ch]] = round(
                    (new_spectrum[ch] - new_noise[ch]), 2
                )  # IEC OSNR for coarsely-distributed channels with measurable noise floors
                # current_osnr[central_freq[ch]] = round(
                #     10*math.log10((db2lin(new_spectrum[ch]) - db2lin(new_noise[ch])) / (db2lin(new_noise[ch])/2)), 
                #     2,
                # )  # pol-mux OSNR estimation for densely-distributed channels with non-measurable noise floors
            else:
                print("Error! very weak channel signal")

        else:
            print("Error! very weak channel signal")
            current_osnr[central_freq[ch]] = round(
                (new_spectrum[ch] - new_noise[ch]), 2
            )

    return current_osnr


def PrintTime(print_flag):
    timestamp = time.localtime()
    pt_time = time.strftime("%Y.%m.%d.%H.%M.%S", timestamp)
    print(print_flag, pt_time)
    return timestamp


def Evaluate(
    line,
    parameters,
    ssh_list,
    estimated_channel_osnr,
    estimated_channel_power,
    delay_vector,
    write_sign,
    metric_sign,
    ssh_flag,
):  ## evaluate the generated parameter and return reward for next acquisition
    ts0 = PrintTime("Evaluation begins Timestamp")
    # clear transponder measurement bin
    if line.fast_slow_version == "slow":
        clean_transponder(line.transponder_box)
    # write the amplifier parameters to the devices
    if line.fast_slow_version == "slow":
        time.sleep(5)
    ts1 = time.localtime()
    ts2 = time.localtime()
    if write_sign:
        ts1 = PrintTime("write_amp begins Timestamp")
        line.write_amp(parameters, ssh_list, ssh_flag)
        ts2 = PrintTime("write_amp ends Timestamp")
    if line.fast_slow_version == "slow":
        time.sleep(15)
    if line.fast_slow_version == "slow":
        metric_bit = [1, 1, 1, 1, 1, 1]
    else:
        metric_bit = [1, 0, 1, 0, 1, 0]

    transponder_Q_set = {}
    transponder_ber_set = {}
    transponder_esnr_set = {}

    # read line OSNR and transponder metric in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        ts3 = PrintTime("get_line_osnr begins Timestamp")
        osnr = executor.submit(
            get_line_osnr,
            line,
            ssh_list,
            estimated_channel_osnr,
            estimated_channel_power,
            plot_flag=1,
            ssh_flag=ssh_flag,
        )
        current_osnr = osnr.result()
        ts5 = PrintTime("get_line_osnr ends Timestamp")
 
    # we want to maximize the min OSNR first, then the OSNR of all channels to be as large as possible
    reward = 0
    if metric_sign == 0:  # maximize for OSNR
        if len(current_osnr) > 0:
            reward = min(current_osnr.values()) * 10000
        for c in current_osnr:
            reward += current_osnr[c]

    elif metric_sign == 1:  # maximize for Q
        if len(transponder_Q_set) > 0:
            reward = min(transponder_Q_set.values()) * 10000
        for c in transponder_Q_set:
            reward += transponder_Q_set[c]

    else:  # minimize for BER
        if len(transponder_ber_set) > 0:
            reward = max(transponder_ber_set.values()) * 10000
        for c in transponder_ber_set:
            reward += transponder_ber_set[c]

    return_reward = {"bow": (reward, None)}
    ts7 = PrintTime("Evaluation finish Timestamp")

    write_delay = time.mktime(ts2) - time.mktime(ts1)
    read_osnr_delay = time.mktime(ts5) - time.mktime(ts3)
    read_transponder_delay = 0  # no need to read transponder here
    total_delay = time.mktime(ts7) - time.mktime(ts0)
    delay_vector.append(write_delay)
    delay_vector.append(read_osnr_delay)
    delay_vector.append(read_transponder_delay)
    delay_vector.append(total_delay)

    return (
        return_reward,
        current_osnr,
        transponder_ber_set,
        transponder_Q_set,
        transponder_esnr_set,
        delay_vector,
    )


def SafeCheck(parameters, amp_type_id, fiber_spans, transponder_fre, verbose):
    print("\033[1;32m Run Safecheck\033[0;0m")
    safecheck_reward = 1
    left_freq = 999999999
    right_freq = 0
    for tx in transponder_fre:
        if transponder_fre[tx] < left_freq:
            left_freq = transponder_fre[tx]
        if transponder_fre[tx] > right_freq:
            right_freq = transponder_fre[tx]
    ts1 = PrintTime("simulator starts Timestamp")
    
    ## run GNPy simulator 10 times and take average
    estimated_channel_osnr = {}
    estimated_channel_power = {}
    for _ in range(10):
        estimated_channel_osnr_current, estimated_channel_power_current = run_simulator(
            parameters,
            amp_type_id,
            fiber_spans,
            left_freq - 37500,
            right_freq + 37500,
            verbose=verbose,
        )  # 4800 GHz C-band spectrum, @75 GHz spacing
        for ch in estimated_channel_osnr_current:
            if len(estimated_channel_osnr):
                estimated_channel_osnr[ch] += estimated_channel_osnr_current[ch]
                estimated_channel_power[ch] += estimated_channel_power_current[ch]
            else:
                estimated_channel_osnr[ch] = estimated_channel_osnr_current[ch]
                estimated_channel_power[ch] = estimated_channel_power_current[ch]
    for ch in estimated_channel_osnr:
        estimated_channel_osnr[ch] = estimated_channel_osnr[ch]/10
    print("GNPy estimated_channel_osnr", estimated_channel_osnr)
    ts2 = PrintTime("simulator ends Timestamp")
    simulator_delay = time.mktime(ts2) - time.mktime(ts1)
    print("simulator_delay", simulator_delay)

    min_estimated_osnr = 999
    for ch in estimated_channel_osnr:
        if min_estimated_osnr > estimated_channel_osnr[ch]:
            min_estimated_osnr = estimated_channel_osnr[ch]
    safecheck_reward = min_estimated_osnr * 10000

    return safecheck_reward, estimated_channel_osnr, estimated_channel_power


def BayesianOptimization(
    parameter_history,
    reward_history,
    osnr_history,
    ber_history,
    q_history,
    esnr_history,
    write_delay_history,
    read_osnr_delay_history,
    read_transponder_delay_history,
    bo_acquisition_delay_history,
    total_delay_history,
    ax_parameters,
    param_constraints,
    fiber_spans,
    amp_type_id,
    line,
    sobol_num,
    gpei_num,
    metric_sign,
    ssh_flag,
    ssh_list,
    user,
    verbose,
):
    """
    The Bayesian Optimization iterative search loop
    """
    ## Ax BO kernal settings
    strategy = GenerationStrategy(
        name="Sobol",
        steps=[
            GenerationStep(model=Models.SOBOL, num_trials=sobol_num),
            GenerationStep(model=Models.GPEI, num_trials=-1),
        ],
    )
    ax_client = AxClient(generation_strategy=strategy)
    ax_client.create_experiment(
        name="bow_test_experiment",
        parameters=ax_parameters,
        objective_name="bow",
        minimize=False, 
        parameter_constraints=param_constraints, 
        experiment_type="network_systems",
    )
    ## adaptive stopping criteria
    current_best = 0
    increase_ratio = 0
    stopping_improvement = 0.05
    stopping_criteria = gpei_num
    stopping = stopping_criteria  # if best reward keeps increase less than 0.05 for 10 trials , stop
    initial_reward = reward_history[0]
    print(reward_history)
    print(
        "The initial reward value (threshold simulator/practical) is",
        initial_reward["bow"][0],
    )

    # start BO iterations
    try:
        print(
            "\033[1;34m ==========\n Start Bayesian Optimization\n ==========\033[0;0m"
        )
        timestamp = time.localtime()
        pt_time = time.strftime("%Y.%m.%d.%H.%M.%S", timestamp)
        print("Timestamp is:", pt_time)
        while stopping > 0:
            ts1 = PrintTime("get_next_trial starts Timestamp")
            parameters, trial_index = ax_client.get_next_trial()
            ts2 = PrintTime("get_next_trial ends Timestamp")
            get_next_trial_delay = time.mktime(ts2) - time.mktime(ts1)
            print("\033[1;31m get_next_trial_delay", get_next_trial_delay, "\033[0;0m")
            delay_vector = [get_next_trial_delay]

            print("\033[1;34m ==== BO loop", trial_index, "====\033[0;0m")
            for x in parameters:
                parameters[x] = round(parameters[x], 2)
            if line.fast_slow_version == "slow":
                print("sleep 10 before evaluating the parameters")
                time.sleep(10)
            # push parameter history
            parameter_history.append(parameters)
            # safety check by the gnpy simulator
            (
                safecheck_reward_value,
                estimated_channel_osnr,
                estimated_channel_power,
            ) = SafeCheck(parameters, amp_type_id, fiber_spans, line.transponder_fre, verbose)
            
            # the BO generated parameters are safe to deploy (min OSNR should be no smaller than initial state)
            if (
                safecheck_reward_value > initial_reward["bow"][0] - 10000
            ):  # assume 1dB margin
                print(
                    "\033[1;32m SafeCheck passed, parameter safe, deploy parameter to network\033[0;0m"
                )
                # evaluate the current parameters generated by BO
                (
                    current_reward,
                    current_osnr,
                    current_ber,
                    current_q,
                    current_esnr,
                    delay_vector,
                ) = Evaluate(
                    line,
                    parameters,
                    ssh_list,
                    estimated_channel_osnr,
                    estimated_channel_power,
                    delay_vector,
                    write_sign=1,
                    metric_sign=metric_sign,
                    ssh_flag=ssh_flag,
                )

                print("current_osnr", current_osnr)
                print("current Q", current_q)
                print("current BER", current_ber)
                print("current ESNR", current_esnr)
                reward_history.append(current_reward)
                osnr_history.append(current_osnr)
                ber_history.append(current_ber)
                q_history.append(current_q)
                esnr_history.append(current_esnr)

                bo_acquisition_delay_history.append(delay_vector[0])
                write_delay_history.append(delay_vector[1])
                read_osnr_delay_history.append(delay_vector[2])
                read_transponder_delay_history.append(delay_vector[3])
                total_delay_history.append(delay_vector[4] + delay_vector[0])

                ts3 = PrintTime("complete_trial starts Timestamp")
                ax_client.complete_trial(
                    trial_index=trial_index, raw_data=current_reward
                )
                ts4 = PrintTime("complete_trial ends Timestamp")
                complete_trial_delay = time.mktime(ts4) - time.mktime(ts3)
                print("complete_trial_delay", complete_trial_delay)

            # the BO generated parameters are NOT safe to deploy
            else:
                print(
                    "\033[1;32m SafeCheck failed, parameter not safe, so feedback simulator response only\033[0;0m"
                )
                bo_acquisition_delay_history.append(delay_vector[0])
                write_delay_history.append(0)
                read_osnr_delay_history.append(0)
                read_transponder_delay_history.append(0)
                total_delay_history.append(0)

                (
                    current_reward,
                    current_osnr,
                    current_ber,
                    current_q,
                    current_esnr,
                    delay_vector,
                ) = Evaluate(
                    line,
                    parameters,
                    ssh_list,
                    estimated_channel_osnr,
                    estimated_channel_power,
                    delay_vector,
                    write_sign=0,
                    metric_sign=metric_sign,
                    ssh_flag=ssh_flag,
                )

                print("current_osnr", current_osnr)
                print("current Q", current_q)
                print("current BER", current_ber)
                print("current ESNR", current_esnr)
                reward_history.append(current_reward)
                osnr_history.append(current_osnr)
                ber_history.append(current_ber)
                q_history.append(current_q)
                esnr_history.append(current_esnr)

                safecheck_reward = {"bow": (safecheck_reward_value, None)}
                ax_client.complete_trial(
                    trial_index=trial_index, raw_data=safecheck_reward
                )

            # BO iteration termination condition: reward does not get improved for STOPPING_CRITERIA
            if safecheck_reward_value > current_best:
                if current_best > 0:
                    increase_ratio = float(
                        (safecheck_reward_value - current_best) / current_best
                    )
                current_best = safecheck_reward_value
                if increase_ratio > stopping_improvement:
                    stopping = stopping_criteria
                elif trial_index > sobol_num:
                    stopping = stopping - 1
            # if it is worse than history best, then gradually close loop
            elif trial_index > sobol_num:
                stopping = stopping - 1

    except (Exception, KeyboardInterrupt) as ex:  # catch *all* exceptions
        print("Exceptions", ex)
        for d in ssh_list:
            line.ssh_cancel_user(user, ssh_list[d])
            ssh_close(ssh_list[d])
            print("Due to exception, Close ssh connection", ssh_list[d], "to device", d)

    # find the best parameters across all BO runs
    best_parameters, values = ax_client.get_best_parameters()
    for x in best_parameters:
        best_parameters[x] = round(best_parameters[x], 2)

    print("best_parameters", best_parameters, values)

    JSON_LOC = "/home/zhizhenzhong/local/database/bo_json/"
    timestamp = time.localtime()
    pt_time = time.strftime("%Y.%m.%d.%H.%M.%S", timestamp)
    JSONFILE = "ax_" + pt_time + ".json"
    jsonfile_name = JSON_LOC + JSONFILE
    ax_client.save_to_json_file(jsonfile_name)

    # config the best parameter, and read output
    (
        safecheck_reward_value,
        estimated_channel_osnr,
        estimated_channel_power,
    ) = SafeCheck(best_parameters, amp_type_id, fiber_spans, line.transponder_fre, verbose)

    delay_vector = [0]  # this is for deploying the best parameter, no BO anymore
    final_reward, final_osnr, final_ber, final_q, final_esnr, delay_vector = Evaluate(
        line,
        best_parameters,
        ssh_list,
        estimated_channel_osnr,
        estimated_channel_power,
        delay_vector,
        write_sign=1,
        metric_sign=0,
        ssh_flag=ssh_flag,
    )

    reward_history.append(final_reward)
    osnr_history.append(final_osnr)
    ber_history.append(final_ber)
    q_history.append(final_q)
    esnr_history.append(final_esnr)
    bo_acquisition_delay_history.append(delay_vector[0])
    write_delay_history.append(delay_vector[1])
    read_osnr_delay_history.append(delay_vector[2])
    read_transponder_delay_history.append(delay_vector[3])
    total_delay_history.append(delay_vector[4] + delay_vector[0])

    ## close all ssh connections
    for d in ssh_list:
        print("cancel user", d)
        line.ssh_cancel_user(user, ssh_list[d])
        ssh_close(ssh_list[d])
        print("Close ssh connection", ssh_list[d], "to device", d)

    return (
        best_parameters,
        parameter_history,
        reward_history,
        osnr_history,
        ber_history,
        q_history,
        esnr_history,
        write_delay_history,
        read_osnr_delay_history,
        read_transponder_delay_history,
        bo_acquisition_delay_history,
        total_delay_history,
    )


def WavelengthReconfiguration(
    ax_parameters,
    param_constraints,
    fiber_spans,
    amp_type_id,
    line,
    sobol_num,
    gpei_num,
    metric_sign,
    random_range,
    ssh_flag,
    verbose,
):
    """
    The entire wavelength reconfiguration process including BO
    """
    parameter_history = []
    reward_history = []
    osnr_history = []
    ber_history = []
    q_history = []
    esnr_history = []
    write_delay_history = []
    read_osnr_delay_history = []
    read_transponder_delay_history = []
    bo_acquisition_delay_history = []
    total_delay_history = []

    # make sure the proprietary_controller is optimized and stay in service
    proprietary_controller_prepareness(
        line.proprietary_controller_info["location_proprietary_controller"],
        line.proprietary_controller_info["proprietary_controller_id"],
        final_state="INSERVICE",
        verbose=verbose,
    )
    time.sleep(20)
    # create scenario where only partial channels present
    WSS_ONOFF_bulkctl(
        line.wss_onoff["node_id"], line.wss_onoff["ONOFF_name"], line.wss_onoff["a_opqstatus"]
    )
    print("before provisioning:")
    time.sleep(20)  # wait for the WSS control command to take effect
    get_line_osnr(
        line,
        ssh_list={},
        estimated_channel_osnr={},
        estimated_channel_power={},
        plot_flag=1,
        ssh_flag=0,
    )

    # optimize proprietary_controller under partial channels, and turn off proprietary_controller
    proprietary_controller_prepareness(
        line.proprietary_controller_info["location_proprietary_controller"],
        line.proprietary_controller_info["proprietary_controller_id"],
        final_state="OUT_OF_SERVICE",
        verbose=verbose,
    )
    print("\033[1;31m Now we emulate a fiber cut event, with proprietary_controller disabled\033[0;0m")
    time.sleep(10)
    print("Clear transponder parameters first")
    clean_transponder(line.transponder_box)
    time.sleep(120)

    # turn channel on to emulate wavelength provisioning
    WSS_ONOFF_bulkctl(
        line.wss_onoff["node_id"], line.wss_onoff["ONOFF_name"], line.wss_onoff["z_opqstatus"]
    )
    print("after provisioning:")

    # decide ssh method
    ssh_list = {}
    user = "0"
    if ssh_flag == 1:
        ## last is the osnr roadm, first is the controller roadm
        ssh_devices = list(line.amplifiers.keys())
        ## setup dedicated ssh sockets for each device
        for d in ssh_devices:
            ssh = ssh_setup(d)
            ssh_list[d] = ssh
            time.sleep(5)  # wait to make sure ssh connection is up
            user = line.ssh_act_user(ssh)
    print("ssh_list", ssh_list)

    initial_gain_results = line.read_amp(line.amplifiers)

    # Read initial parameter status and store in the correct form
    initial_gain_results_float = []
    initial_gain_results_key = []
    for x in initial_gain_results:
        initial_gain_results_float.append(float(initial_gain_results[x]))
        initial_gain_results_key.append(x)
    initial_gain_results_zip = dict(
        zip(initial_gain_results_key, initial_gain_results_float)
    )
    parameter_history.append(initial_gain_results_zip)

    # initial reward value
    (
        current_reward,
        current_osnr,
        current_ber,
        current_q,
        current_esnr,
        delay_vector,
    ) = Evaluate(
        line,
        initial_gain_results,
        ssh_list,
        estimated_channel_osnr={},
        estimated_channel_power={},
        delay_vector=[0],
        write_sign=0,
        metric_sign=metric_sign,
        ssh_flag=ssh_flag,  # use FCR
    )
    print("initial reward", current_reward)
    print("initial osnr", current_osnr)
    print("initial Q", current_q)
    print("initial BER", current_ber)
    print("initial ESNR", current_esnr)
    reward_history.append(current_reward)
    osnr_history.append(current_osnr)
    ber_history.append(current_ber)
    q_history.append(current_q)
    esnr_history.append(current_esnr)

    bo_acquisition_delay_history.append(delay_vector[0])
    write_delay_history.append(delay_vector[1])
    read_osnr_delay_history.append(delay_vector[2])
    read_transponder_delay_history.append(delay_vector[3])
    total_delay_history.append(delay_vector[4] + delay_vector[0])

    (
        best_parameters,
        parameter_history,
        reward_history,
        osnr_history,
        ber_history,
        q_history,
        esnr_history,
        write_delay_history,
        read_osnr_delay_history,
        read_transponder_delay_history,
        bo_acquisition_delay_history,
        total_delay_history,
    ) = BayesianOptimization(
        parameter_history,
        reward_history,
        osnr_history,
        ber_history,
        q_history,
        esnr_history,
        write_delay_history,
        read_osnr_delay_history,
        read_transponder_delay_history,
        bo_acquisition_delay_history,
        total_delay_history,
        ax_parameters,
        param_constraints,
        fiber_spans,
        amp_type_id,
        line,
        sobol_num,
        gpei_num,
        metric_sign,
        ssh_flag,
        ssh_list,
        user,
        verbose,
    )

    return (
        best_parameters,
        parameter_history,
        reward_history,
        osnr_history,
        ber_history,
        q_history,
        esnr_history,
        write_delay_history,
        read_osnr_delay_history,
        read_transponder_delay_history,
        bo_acquisition_delay_history,
        total_delay_history,
    )
