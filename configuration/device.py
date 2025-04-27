import socket
import psutil
import sys
from configuration.registration import server_data


def get_ip_address(ifname):
    if sys.platform == "win32":
        # Windows 获取所有网卡信息
        addrs = psutil.net_if_addrs()
        for interface, addrs_list in addrs.items():
            if interface.lower() == ifname.lower():
                for addr in addrs_list:
                    if addr.family == socket.AF_INET:
                        return addr.address
    else:
        # Linux 获取网卡的IP地址（原代码方法）
        import fcntl
        import struct
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            info = fcntl.ioctl(s.fileno(), 0x8915, struct.pack('256s', bytes(ifname[:15], 'utf-8')))
            return socket.inet_ntoa(info[20:24])
        except Exception as e:
            raise Exception(f"Error obtaining IP address for {ifname}: {str(e)}")

    # s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # info = fcntl.ioctl(s.fileno(), 0x8915, struct.pack('256s', bytes(ifname[:15], 'utf-8')))
    # return socket.inet_ntoa(info[20:24])

def assign_service(moda='eno1'):
    # moda: eno1, lo
    ip = get_ip_address(moda)
    root_path = server_data[ip]

    return ip, root_path

