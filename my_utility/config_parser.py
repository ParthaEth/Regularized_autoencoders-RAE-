def get_config_idxs(process_idx, config_dict):
    maj_cfg_idx = 0
    count = 0
    while(True):
        num_minor_configs = len(config_dict[maj_cfg_idx])-1
        if count + num_minor_configs >= process_idx + 1:
            minor_cfg_idx = process_idx - count
            break
        else:
            count += num_minor_configs
            maj_cfg_idx += 1
    return maj_cfg_idx, minor_cfg_idx + 1


def get_process_id_given_mj_minor_idxs(major_cfg_idx_target, minor_cfg_idx_target, config_dict):
    process_id = 0
    for major_cfg_idx in range(major_cfg_idx_target):
        num_minor_configs = len(config_dict[major_cfg_idx]) - 1
        process_id += num_minor_configs
    process_id += minor_cfg_idx_target
    return process_id-1


if __name__ == "__main__":
    import sys
    sys.path.append('./')
    sys.path.append('../')
    from configurations import config

    mj_idx = int(sys.argv[1])
    min_idx = int(sys.argv[2])
    print("PID = " + str(get_process_id_given_mj_minor_idxs(mj_idx, min_idx, config.configurations)))