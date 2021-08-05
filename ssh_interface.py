import subprocess


def ssh_setup(host_name):
    host_name = host_name + "-1"
    ssh = subprocess.Popen(
        ["ssh", "-i .ssh/id_rsa", host_name],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=0,
    )
    print("Setting direct ssh to", host_name)
    return ssh


def ssh_write(ssh, cmd):
    print("* ssh write", cmd)
    try:
        ssh.stdin.write(cmd)
    except Exception as ex:
        print("Exceptions", ex)
        ssh_close(ssh)


def ssh_read(ssh, identify_sign):
    print("* ssh read begins")
    results = []
    last_identify = "NA"
    for line in ssh.stdout:
        if identify_sign in line.strip():
            results.append(line.strip())
            last_identify = identify_sign
        else:
            last_identify = "NA"

        if line.strip() == ";" and last_identify == identify_sign:
            break
        if len(results):
            if "196125000" in results[-1]:  # test spectrum end frequency
                break

    print("* ssh read ends")
    return results


def ssh_close(ssh):
    ssh.stdin.close()
    ssh.stdout.close()
    ssh.stderr.close()
    ssh.kill()
    print("close ssh connection")
