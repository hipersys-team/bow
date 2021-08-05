import sys

from bayesian_amp_control import (
    WavelengthReconfiguration,
    Evaluate,
    SafeCheck,
)
from input_parameters import (
    amplifiers_location,
    wss_location,
    proprietary_controller_info,
    transponder_box,
    wss_onoff,
    fiber_spans,
    plots_location,
    param_constraints,
    ax_parameters,
    amp_type_id,
)
from plotting import line_plot, bar_plot, violin_plot
from fcr_interface import Line_Device, get_transponder_adj


def run(
    plots_location,
    amplifiers_location,
    wss_location,
    wss_onoff,
    transponder_box,
    proprietary_controller_info,
    param_constraints,
    ax_parameters,
    sobol_num,
    gpei_num,
    metric_sign,
    random_range,
    fiber_spans,
    amp_type_id,
    fast_slow_version,
    ssh_flag,
    verbose,
):
    print("\033[1;34m ==static query to know the setting of transponders==\033[0;0m")
    
    # specify transponders used for our experiment
    transponder_mod, transponder_fre = get_transponder_adj(
        transponder_box, metric_bit=[1, 1]
    )
    print("transponder_mod", transponder_mod)
    print("transponder_fre", transponder_fre)

    # specify line devices used for our experiment
    line = Line_Device(
        amplifiers_location,
        wss_location,
        wss_onoff,
        proprietary_controller_info,
        transponder_box,
        transponder_fre,
        fast_slow_version,
    )  
    print("line devices", line)

    ## run Bayesian Optimization loop
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
    ) = WavelengthReconfiguration(
        ax_parameters,
        param_constraints,
        fiber_spans,
        amp_type_id,
        line,
        sobol_num=sobol_num,
        gpei_num=gpei_num,
        metric_sign=metric_sign,
        random_range=random_range,
        ssh_flag=ssh_flag,
        verbose=verbose,
    )

    # print the parameter history during bO
    print("osnr_history", osnr_history)
    print("ber_history", ber_history)
    print("q_history", q_history)
    print("esnr_history", esnr_history)

    # figure out the BO index of the best parameter
    best_index = 0
    for y in range(len(parameter_history)):
        best_index_flag = 0
        for x in best_parameters:
            if best_parameters[x] != parameter_history[y][x]:
                best_index_flag = 1
                break
        if best_index_flag == 0:
            best_index = y
            break

    # final process osnr_history
    for k in range(len(osnr_history)):
        osnr_keys = []
        if len(osnr_history[k]) == 0:
            for x in osnr_keys:
                osnr_history[k][x] = 0
        else:
            osnr_keys = list(osnr_history[k].keys())

    # plotting results for visualization
    sobol_index = sobol_num
    line_plot(
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
    )
    bar_plot(
        plots_location,
        write_delay_history,
        read_osnr_delay_history,
        read_transponder_delay_history,
        bo_acquisition_delay_history,
        total_delay_history,
    )

    return (
        best_parameters,
        best_index,
        parameter_history,
        reward_history,
        osnr_history,
        ber_history,
        q_history,
        esnr_history,
    )


if __name__ == "__main__":
    print("\033[0;33m ================================================\n\033[0;0m")
    print("\033[1;33m ######  ####### #     #\033[0;0m")
    print("\033[1;33m #     # #     # #  #  #\033[0;0m")
    print("\033[1;33m ######  #     # #  #  #\033[0;0m")
    print("\033[1;33m #     # #     # #  #  #\033[0;0m")
    print("\033[1;33m #     # #     # #  #  #\033[0;0m")
    print("\033[1;33m ######  #######  ## ##\n\033[0;0m")
    print("\033[1;34m BOW: Bayesian-Optimized Wavelengths\n\033[0;0m")
    print("\033[0;34m [Paper]: Z. Zhong, M. Ghobadi, M. Balandat, S. Katti, A. Kazerouni, J. Leach, M. McKillop, Y. Zhang, BOW: First Real-World Demonstration of a Bayesian Optimization System for Wavelength Reconfiguration, OFC 2021 (Postdeadline Paper).\033[0;0m")
    print("\033[0;34m [Website]: http://bow.csail.mit.edu\033[0;0m")
    print("\033[0;34m [Code Contributor]: zhizhenz@mit.edu\033[0;0m")
    print("\033[0;34m [Code Release Date]: August 4, 2021\033[0;0m")
    print("\033[0;33m================================================\n\033[0;0m")
    

    (
        best_parameters,
        best_index,
        parameter_history,
        reward_history,
        osnr_history,
        ber_history,
        q_history,
        esnr_history,
    ) = run(
        plots_location,
        amplifiers_location,
        wss_location,
        wss_onoff,
        transponder_box,
        proprietary_controller_info,
        param_constraints,
        ax_parameters,
        sobol_num=10,
        gpei_num=10,
        metric_sign=0,
        random_range=0,
        fiber_spans=fiber_spans,
        amp_type_id=amp_type_id,
        fast_slow_version="fast",
        ssh_flag=0,  # 0 for FCR, 1 for direct SSH
        verbose=True,
    )

    sys.exit()
