## network details
amplifiers_location = {
    "amp1": "AMP-1",
    "amp2": "AMP-1",
    "amp3": "AMP-1",
    "amp4": "AMP-1",
    "amp5": "AMP-1",
    "amp6": "AMP-1",
    "amp7": "AMP-1",
    "amp8": "AMP-1",
}
amp_typeid = {
    "amp1": "high_detail_model_example",
    "amp2": "high_detail_model_example",
    "amp3": "high_detail_model_example",
    "amp4": "high_detail_model_example", 
    "amp5": "high_detail_model_example", 
    "amp6": "high_detail_model_example", 
    "amp7": "high_detail_model_example", 
    "amp8": "high_detail_model_example",
}
## this is the WSS at the end of the rail where we can read spectrum
wss_location = {
    "roadm_id": "roadm",
    "chassis_id": "x",
    "card_id": "x",
    "port_id": "x",
    "channel_num": 12,
    "grid": 24,
    "start_freq": 193012500,
}
## this is the proprietary_controller location which is responsible for managing this rail in the source ROADM site
proprietary_controller_info = {
    "location_proprietary_controller": "x",
    "proprietary_controller_id": "x",
}
transponder_box = {
    "transponder1": ["1/1", "1/2", "2/1", "2/2"],
    "transponder2": ["1/1", "1/2", "2/1", "2/2"],
}
wss_onoff = {
    "node_id": "x",
    "channel_name": [
        "CH1",  # 194.25-194.325
        "CH2",  # 194.1-194.175
        "CH3",  # 193.95-194.025
        "CH4",  # 193.8-193.875
        "CH5",  # 193.65-193.725
        "CH6",  # 193.35-193.425
        "CH7",  # 193.2-193.275
        "CH8",  # 193.05-193.125
    ],
    "a_opqstatus": [
        "OPQ=YES",
        "OPQ=YES",
        "OPQ=YES",
        "OPQ=YES",
        "OPQ=YES",
        "OPQ=YES",
        "OPQ=YES",
        "OPQ=NO",
    ],
    "z_opqstatus": [
        "OPQ=NO",
        "OPQ=NO",
        "OPQ=NO",
        "OPQ=NO",
        "OPQ=NO",
        "OPQ=NO",
        "OPQ=NO",
        "OPQ=NO",
    ],
}
channel = {
    "name": [
        "channel0-1-1",
        "channel0-1-2",
        "channel0-1-3",
        "channel0-1-4",
        "channel0-1-5",
        "channel0-1-6",
        "channel0-1-7",
        "channel0-1-8",
    ],
    "proprietary_controller_node": "x",
    "a_status": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
    ],
    "z_status": [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ],
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
        "name": "amp5=anmp",
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
# constraints for gain of all amplifiers
param_constraints = [
    "amp1 + amp2 + amp3 + amp4 + amp5 + amp6 + amp7 + amp8 <= 134",
    "amp1 + amp2 + amp3 + amp4 + amp5 + amp6 + amp7 + amp8 >= 128",
]

ax_parameters = [
    {
        "name": "amp1",
        "type": "range",
        "bounds": [12, 16],
        "value_type": "float",
        "digits": 1,
    },
    {
        "name": "amp2",
        "type": "range",
        "bounds": [13.5, 17.5],
        "value_type": "float",
        "digits": 1,
    },
    {
        "name": "amp3",
        "type": "range",
        "bounds": [16, 20],
        "value_type": "float",
        "digits": 1,
    },
    {
        "name": "amp4",
        "type": "range",
        "bounds": [18, 22], 
        "value_type": "float",
        "digits": 1,
    },
    {
        "name": "amp5",
        "type": "range",
        "bounds": [17, 21],
        "value_type": "float",
        "digits": 1,
    },
    {
        "name": "amp6",
        "type": "range",
        "bounds": [13, 17], 
        "value_type": "float",
        "digits": 1,
    },
    {
        "name": "amp7",
        "type": "range",
        "bounds": [14.5, 18.5], 
        "value_type": "float",
        "digits": 1,
    },
    {
        "name": "amp8",
        "type": "range",
        "bounds": [12, 16],  
        "value_type": "float",
        "digits": 1,
    },
]

plots_location = "/home/zhizhenzhong/local/database/ofc21_plots/"
ampmonitor_location = "/home/zhizhenzhong/local/database/ampmonitor/"
