import copy
import datetime
import json
import logging
import re
import time
from collections import defaultdict

import pyjq
from crypto.keychain_service.keychain import ttypes as keychain
from fbnet.command_runner.CommandRunner import ttypes as fcr_ttypes
from libfb.py.thrift_clients.fbnet_command_runner_thrift_client import (
    FBNetCommandRunnerThriftClient as Legacy_FcrClient,
)
from libfb.py.thrift_clients.keychain_thrift_client import KeychainClient
from scripts.neteng.optical.bow.ssh_interface import (
    ssh_read,
    ssh_write,
)

# define the class of line devices
class Line_Device(object):
    def __init__(
        self,
        amplifiers,
        wss,
        wss_onoff,
        proprietary_controller_info,
        transponder_box,
        transponder_fre,
        fast_slow_version,
    ):
        self.amplifiers = amplifiers
        self.wss = wss
        self.wss_onoff = wss_onoff
        self.proprietary_controller_info = proprietary_controller_info
        self.transponder_box = transponder_box
        self.transponder_fre = transponder_fre
        self.fast_slow_version = fast_slow_version

    # system password for authetication
    def get_secret(self, name, group):
        req = keychain.GetSecretRequest(
            name=name,
            group=group,
        )
        try:
            return KeychainClient().getSecret(req)
        except keychain.KeychainServiceException as ex:
            print("Error retrieving secret:" + ex)
            return False

    # activate user on device
    def ssh_act_user(self, ssh):
        print("Activating ssh user\n")
        user = self.get_secret("TACACS_USERNAME", "NETENG_AUTOMATION").secret
        pw = self.get_secret("TACACS_PASSWORD", "NETENG_AUTOMATION").secret
        user_add_cmd = "XXXXXXXXXXX" + user + pw   ## anonymized southbound command
        ssh_write(ssh, user_add_cmd)
        results = ssh_read(ssh, identify_sign="")
        print("ssh_act_user results", results, "\n")
        return user

    def ssh_cancel_user(self, user, ssh):
        print("Canceling ssh user\n")
        user_cancel_cmd = "XXXXXXXXXXX" + user  ## anonymized southbound command
        ssh_write(ssh, user_cancel_cmd)
        results = ssh_read(ssh, identify_sign="")
        print("ssh_cancel_user resutls", results, "\n")

    def tl1_ssh_runner(self, ssh, cmd, identify_sign):
        ssh_write(ssh, cmd)
        results = ssh_read(ssh, identify_sign=identify_sign)
        micro_timestamp = datetime.datetime.now()
        return results, micro_timestamp

    def tl1_device_bulkrunner(self, device_commands, read_or_write):
        try:
            user = self.get_secret("TACACS_USERNAME", "NETENG_AUTOMATION").secret
            pw = self.get_secret("TACACS_PASSWORD", "NETENG_AUTOMATION").secret
            if read_or_write == 1:
                persistent_sign = "auto"
            else:
                persistent_sign = "auto"

            fcr_commands = {
                fcr_ttypes.Device(
                    hostname=device,
                    username=user,
                    password=pw,
                    session_type=fcr_ttypes.SessionType.TL1,
                    session_data=fcr_ttypes.SessionData(
                        extra_options={
                            "format": "json",
                            "use_persistent_connection": persistent_sign,
                        }
                    ),
                ): device_commands[device]
                for device in device_commands
            }
            with Legacy_FcrClient() as client:
                res = client.bulk_run(fcr_commands)
                timestamp = time.localtime()
                return res, timestamp

        except Exception as ex:
            print("User exception: {}".format(ex))
            timestamp = time.localtime()
            return 1, timestamp

    # get the spectrum peaks of the channels on the testbed, vacant wavelength is 0
    def read_spectrum(self, wss, ssh_list, ssh_flag):
        spectrum_spectrum = []
        noise_spectrum = []
        central_freq = []
        corrected_central_freq = []
        fine_grain_spectrum = []
        fine_grain_frequency = []
        bins = []

        roadm_id = wss["roadm_id"]
        chassis_id = wss["chassis_id"]
        card_id = wss["card_id"]
        port_id = wss["port_id"]
        max_channel_read_num = wss["channel_num"]
        grid_space = wss["grid"]
        startchannel_freq = wss["start_freq"]

        if ssh_flag == 1:  # use ssh one-time authentication
            print("\033[1;31m\n** Use SSH direct connection for read_spectrum\033[0;0m")
            command = "XXXXXXXXXXX"  ## anonymized southbound command

            results, micro_timestamp = self.tl1_ssh_runner(
                ssh_list[roadm_id], command, identify_sign="XXXXXXXXXXX"
            )
            print("tl1_ssh_runner", results, "\n")
            channel_id = []
            frequency = []
            power = []
            for s in results:
                if s.startswith('"spectrum'):
                    channel_id.append(str(s[1:-1].split(",")[0]))
                    frequency.append(int(s[1:-1].split(",")[0].split("-")[-1]))
                    power.append(float(s[1:-1].split(",")[2]))

        else:  # use FCR
            print("\033[1;31m\n** Use FCR client connection for read_spectrum\033[0;0m")
            command = "XXXXXXXXXXX"  ## anonymized southbound command

            commands = defaultdict(list)
            commands[roadm_id].append(command)

            rc = 1
            count = 0
            while rc == 1 and count < 10:  # rc = 1 means exception
                rc, timestamp = self.tl1_device_bulkrunner(commands, read_or_write=1)
                count += 1

            if rc == 1:
                print("the FCR fails to fetch data\n")
                channel_id = []
                frequency = []
                power = []
            else:
                j_data = json.loads(rc[roadm_id][0].output)
                vars = {}
                jqscript = ".[][].fields | {keys: .[0][0], value: . [1][1]}"
                results = pyjq.all(jqscript, j_data, vars=vars)

                channel_id = list(map(lambda x: str(x["keys"]), results))
                frequency = list(map(lambda x: int(x["keys"].split("-")[-1]), results))
                power = list(map(lambda x: float(x["value"]), results))

        spectrum_reading = {channel_id[i]: power[i] for i in range(len(frequency))}
        # print(spectrum_reading)
        loc_match = "spectrum-" + chassis_id + "-" + card_id + "-" + port_id
        # print(loc_match)

        step = 0
        channel_num = 0
        current_fre_bin = []
        current_power_bin = []

        for item in range(len(spectrum_reading)):
            if channel_id[item].startswith(loc_match):
                fine_grain_spectrum.append(power[item])
                fine_grain_frequency.append(channel_id[item])
                if frequency[item] >= start_freq + grid_space * channel_num:
                    # print(frequency[item])
                    current_fre_bin.append(frequency[item])
                    current_power_bin.append(power[item])
                    step = step + 1

                    if step == grid_space:
                        peak_power = max(current_power_bin)
                        loc = current_power_bin.index(peak_power)
                        noise = float(
                            (
                                current_power_bin[0]
                                + current_power_bin[1]
                                + current_power_bin[-1]
                                + current_power_bin[-2]
                            )
                            / 4
                        )

                        if peak_power < -15:
                            pass
                        else:
                            spectrum_spectrum.append(peak_power)
                            central_freq.append(current_fre_bin[loc])
                            noise_spectrum.append(noise)
                            current_fre_bin_copy = copy.deepcopy(current_fre_bin)
                            bins.append(current_fre_bin_copy)

                        current_fre_bin.clear()
                        current_power_bin.clear()
                        channel_num = channel_num + 1
                        step = 0

                        if channel_num == max_channel_read_num:
                            break

        ## if the detected central frequency is closest to one of the transponder frequency
        for c in central_freq:
            for x in self.transponder_fre:
                if (
                    abs(c - self.transponder_fre[x]) < 75000
                ):  # frequency slot size is 75 GHz
                    corrected_central_freq.append(self.transponder_fre[x])
                    break

        return (
            spectrum_spectrum,
            noise_spectrum,
            corrected_central_freq,
            fine_grain_spectrum,
            fine_grain_frequency,
            bins,
        )

    def write_amp(self, parameters, ssh_list, ssh_flag):
        write_amp_command = "XXXXXXXXXXX" ## anonymized southbound command
        if ssh_flag:
            print("\033[1;31m\n** Use SSH direct connection for write_amp\033[0;0m")
            for amp in ssh_list:
                print("write to", amp, "via ssh", ssh_list[amp])
                gain_command = (
                    write_amp_command.format(amplifier_name=self.amplifiers[amp])
                    + str(parameters[amp])
                    + ";"
                )
                rc, micro_timestamp = self.tl1_ssh_runner(
                    ssh_list[amp], gain_command, identify_sign=""
                )
                print("tl1_ssh_runner", rc, "\n")
        else:
            print("\033[1;31m\n** Use FCR for write_amp\033[0;0m")
            commands = defaultdict(list)
            for amp in parameters:
                gain_command = (
                    write_amp_command.format(amplifier_name=self.amplifiers[amp])
                    + str(parameters[amp])
                    + ";"
                )
                commands[amp].append(gain_command)

            rc = 1
            count = 0
            while rc == 1 and count < 10:  # rc = 1 means exception
                rc, timestamp = self.tl1_device_bulkrunner(commands, read_or_write=0)
                count += 1

    def read_amp(self, amplifiers):
        query_amplifier_cmd = "XXXXXXXXXXX" ## anonymized southbound command
        commands = defaultdict(list)
        for amp in amplifiers:
            commands[amp].append(query_amplifier_cmd.format(amp_name=amplifiers[amp]))
        print(commands)

        results = 1
        count = 0
        while results == 1 and count < 10:  # rc = 1 means exception
            results, timestamp = self.tl1_device_bulkrunner(commands, read_or_write=1)
            count += 1

        gain_results = {}
        if results == 1:
            print("the FCR fails to fetch data")
        else:
            for device, command_results in results.items():
                for command_result in command_results:
                    if command_result.status != "success":
                        logging.error(f"{command_result.command} failed on {device}")
                        continue
                    gain = self.process_AMP(command_result)
                    gain_results[device] = gain

        return gain_results

    def process_AMP(self, rc):
        j_data = json.loads(rc.output)
        vars = {}
        jqscript = ".[][].fields | {keys: .[0], value: .[2]}"
        results = pyjq.all(jqscript, j_data, vars=vars)
        gain = 0
        for i in range(len(results)):
            if results[i]["value"]["AMPMODE"] == "GAINCLAMP":
                gain = results[i]["value"]["GAIN"]

        return gain


# query transponder 
def bulkrun_cli_command(device_commands, username, password):
    fcr_commands = {
        fcr_ttypes.Device(
            hostname=device,
            username=username,
            password=password,
            session_type=fcr_ttypes.SessionType.TL1,
            session_data=fcr_ttypes.SessionData(
                extra_options={
                    "format": "json",
                    "use_persistent_connection": "auto",
                }
            ),
        ): device_commands[device]
        for device in device_commands
    }

    with Legacy_FcrClient() as client:
        res = client.bulk_run(fcr_commands)
        return res


# bulk query
def bulk_query_cli(device_commands):
    try:
        username = self.get_secret("TACACS_USERNAME", "NETENG_AUTOMATION").secret
        ppasswordw = self.get_secret("TACACS_PASSWORD", "NETENG_AUTOMATION").secret
        response = bulkrun_cli_command(
            device_commands=device_commands,
            username=username,
            password=password,
        )
        timestamp = time.localtime()
        return response, timestamp

    except Exception as ex:
        print("exception: {}".format(ex))


# clean the transponder collection bin
def clean_transponder(transponder_box):
    ts_command_set = [
        "XXXXXXXXXXX" ## anonymized southbound command
    ]
    commands = defaultdict(list)

    ## construct commands for bulk run
    for ts_box in transponder_box:
        for transponder in transponder_box[ts_box]:
            clear_command = ts_command_set[0] + transponder
            commands[ts_box].append(clear_command)

    print("commands", commands)
    rc, timestamp = bulk_query_cli(commands)


# query transponders adjacency
def get_transponder_adj(transponder_box, metric_bit):
    ts_command_set = [
        "XXXXXXXXXXX" ## anonymized southbound command
    ]
    commands = defaultdict(list)

    ## construct commands for bulk run
    for ts_box in transponder_box:
        for transponder in transponder_box[ts_box]:
            Modulation = "N/A"
            Frequency = 0
            query_command = ts_command_set[0] + transponder  ## query performance metric
            commands[ts_box].append(query_command)

    print("get_transponder_adj commands", commands)
    rc, timestamp = bulk_query_cli(commands)

    transponder_mod = {}
    transponder_fre = {}

    for tx_box in transponder_box:
        for tx in range(len(commands[tx_box])):
            if metric_bit[0]:
                regex_mod = r"Modulation Scheme"  ## anonymized regex
                matches_mod = re.finditer(
                    regex_mod, rc[tx_box][tx].output, re.MULTILINE
                )
                for _, match in enumerate(matches_mod, start=1):
                    for groupNum in range(0, len(match.groups())):
                        groupNum = groupNum + 1
                        Modulation = str(match.group(groupNum))

            if metric_bit[1]:
                regex_fre = r"Frequency \(GHz\)"  ## anonymized regex
                matches_fre = re.finditer(
                    regex_fre, rc[tx_box][tx].output, re.MULTILINE
                )
                for _, match in enumerate(matches_fre, start=1):
                    for groupNum in range(0, len(match.groups())):
                        groupNum = groupNum + 1
                        if match.group(groupNum) == "N/A":
                            Frequency = 999
                        else:
                            Frequency = int(float(match.group(groupNum)) * 1000)

            transponder_mod[tx_box + "-" + transponder_box[tx_box][tx]] = Modulation
            transponder_fre[tx_box + "-" + transponder_box[tx_box][tx]] = Frequency

    return transponder_mod, transponder_fre


## query transponders performance metric
def get_transponder_pm(line, ssh_list, metric_bit, ssh_flag):
    clean_transponder(line.transponder_box)
    time.sleep(5)
    ts_command_set = [
        "XXXXXXXXXXX" ## anonymized southbound command
    ]
    if ssh_flag == 1:  # use ssh one-time authentication
        print(
            "\033[1;31m\n** Use SSH direct connection for get_transponder_pm\033[0;0m"
        )
        for ts_box in line.transponder_box:
            for transponder in line.transponder_box[ts_box]:
                BER = None
                BERMax = None
                Qfactor = None
                QfactorMin = None
                ESNR = None
                ESNRmin = None
                query_command = (
                    ts_command_set[0]
                    + transponder
                    + " bin-type untimed"  # untimed bin
                )  ## query performance metric

                rc, micro_timestamp = line.tl1_ssh_runner(
                    ts_box, query_command, identify_sign=""
                )
                print("tl1_ssh_runner", rc, "\n")

    else:  # use FCR
        print("\033[1;31m\n** Use FCR for get_transponder_pm\033[0;0m")
        commands = defaultdict(list)
        ## construct commands for bulk run
        for ts_box in line.transponder_box:
            for transponder in line.transponder_box[ts_box]:
                BER = None
                BERMax = None
                Qfactor = None
                QfactorMin = None
                ESNR = None
                ESNRmin = None
                query_command = (
                    ts_command_set[0]
                    + transponder
                    + " bin-type untimed"  # untimed bin
                )  ## query performance metric
                commands[ts_box].append(query_command)

        print("commands", commands)
        rc, timestamp = bulk_query_cli(commands)

    transponder_Q_set = {}
    transponder_Qmin_set = {}
    transponder_ber_set = {}
    transponder_bermax_set = {}
    transponder_esnr_set = {}
    transponder_esnrmin_set = {}

    for tx_box in line.transponder_box:
        for tx in range(len(commands[tx_box])):
            BER = None
            BERMax = None
            Qfactor = None
            QfactorMin = None
            ESNR = None
            ESNRmin = None

            if metric_bit[0]:
                regex_q = r"Q-factor"  ## anonymized regex
                matches_q = re.finditer(regex_q, rc[tx_box][tx].output, re.MULTILINE)
                for _, match in enumerate(matches_q, start=1):
                    for groupNum in range(0, len(match.groups())):
                        groupNum = groupNum + 1
                        if match.group(groupNum) == "N/A":
                            Qfactor = None
                        else:
                            Qfactor = float(match.group(groupNum))

            if metric_bit[1]:
                regex_qmin = r"Q-factor Min"  ## anonymized regex
                matches_qmin = re.finditer(
                    regex_qmin, rc[tx_box][tx].output, re.MULTILINE
                )
                for _, match in enumerate(matches_qmin, start=1):
                    for groupNum in range(0, len(match.groups())):
                        groupNum = groupNum + 1
                        if match.group(groupNum) == "N/A":
                            QfactorMin = None
                        else:
                            QfactorMin = float(match.group(groupNum))

            if metric_bit[2]:
                regex_ber = r"Pre-FEC BER"  ## anonymized regex
                matches_ber = re.finditer(
                    regex_ber, rc[tx_box][tx].output, re.MULTILINE
                )
                for _, match in enumerate(matches_ber, start=1):
                    for groupNum in range(0, len(match.groups())):
                        groupNum = groupNum + 1
                        if match.group(groupNum) == "N/A":
                            BER = None
                        else:
                            BER = float(match.group(groupNum))

            if metric_bit[3]:
                regex_bermax = r"Pre-FEC BER Max"  ## anonymized regex
                matches_bermax = re.finditer(
                    regex_bermax, rc[tx_box][tx].output, re.MULTILINE
                )
                for _, match in enumerate(matches_bermax, start=1):
                    for groupNum in range(0, len(match.groups())):
                        groupNum = groupNum + 1
                        if match.group(groupNum) == "N/A":
                            BERMax = None
                        else:
                            BERMax = float(match.group(groupNum))

            if metric_bit[4]:
                regex_ESNR = r"ESNR Avg"  ## anonymized regex
                matches_bermax = re.finditer(
                    regex_ESNR, rc[tx_box][tx].output, re.MULTILINE
                )
                for _, match in enumerate(matches_bermax, start=1):
                    for groupNum in range(0, len(match.groups())):
                        groupNum = groupNum + 1
                        if match.group(groupNum) == "N/A":
                            ESNR = None
                        else:
                            ESNR = float(match.group(groupNum))

            if metric_bit[5]:
                regex_ESNRmin = r"ESNR Min"  ## anonymized regex
                matches_bermax = re.finditer(
                    regex_ESNRmin, rc[tx_box][tx].output, re.MULTILINE
                )
                for _, match in enumerate(matches_bermax, start=1):
                    for groupNum in range(0, len(match.groups())):
                        groupNum = groupNum + 1
                        if match.group(groupNum) == "N/A":
                            ESNRmin = None
                        else:
                            ESNRmin = float(match.group(groupNum))

            frequency = line.transponder_fre[
                tx_box + "-" + line.transponder_box[tx_box][tx]
            ]
            transponder_Q_set[frequency] = Qfactor
            transponder_Qmin_set[frequency] = QfactorMin
            transponder_ber_set[frequency] = BER
            transponder_bermax_set[frequency] = BERMax
            transponder_esnr_set[frequency] = ESNR
            transponder_esnrmin_set[frequency] = ESNRmin

    return (
        transponder_Q_set,
        transponder_Qmin_set,
        transponder_ber_set,
        transponder_bermax_set,
        transponder_esnr_set,
        transponder_esnrmin_set,
    )
