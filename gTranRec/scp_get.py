import os
from paramiko import SSHClient
from scp import SCPClient
import sys
from datetime import date
from datetime import timedelta
import time
import pkg_resources
from urllib.request import urlretrieve
import gotocat as gc


def progress(filename, size, sent):
        """
        To show the progress of the downloading file.
        """
        sys.stdout.write("%s\'s progress: %.2f%%   \r" % (filename, float(sent)/float(size)*100) )
    
def ssh_scp_get(ssh_host, ssh_user, ssh_password, ssh_port, remote_path, local_path):
        """
        To connect the remove host through SSH and download the file from the given remote path to the local path.
        ***
        ssh_host: str
                IP of the remote host
        ssh_user: str
                username in the host
        ssh_password: str
                password for logging in
        ssh_port: str

        remote_path: str
                path of the parent directory of the file in the remote host
        local_path: str
                path in the local machine you want the file to be downloaded
        """
        ssh = SSHClient()
        ssh.load_system_host_keys()
        ssh.connect(ssh_host, username=ssh_user, password=ssh_password, look_for_keys=False)

        with SCPClient(ssh.get_transport(), progress=progress) as scp:
                scp.get(remote_path, local_path)

def pull(username, password, filename, phase=4):
        """
        Run the 'ssh_scp_get'.
        ***
        username: str
                username logging in 'gotohead'
        password: str
                password logging in 'gotohead'
        obsdate: str
                'yyyy-mm-dd' of the observation taken
        filename: str
                filename of the FITS you want to download (eg. r0123456_UT1-median.fits)
        """

        g4 = gc.GOTOdb(phase=phase)
        file_info = g4.query("SELECT * FROM image WHERE filename='{}'".format(filename)).iloc[0]
        obsdate = str(file_info['obsdate']).split(' ')[0].split("-")
        obsdate = date(int(obsdate[0]), int(obsdate[1]), int(obsdate[2]))
        host = "goto-observatory.warwick.ac.uk"
        image_path = ["/export/gotodata1/gotophoto/storage/pipeline/", "/export/gotodata2/gotophoto/storage/pipeline/"]
        for path in image_path:
                if not os.path.isfile(filename):
                        try:
                                try:
                                        file_path = path + str(obsdate) + "/final/"
                                        print("Trying to get {} from {}".format(filename, file_path))
                                        ssh_scp_get(host, username, password, None, file_path + filename, "./")
                                except:
                                        dt = timedelta(days = 1)
                                        old_obsdate = obsdate - dt
                                        file_path = path + str(old_obsdate) + "/final/"
                                        print("Trying to get {} from {}".format(filename, file_path))
                                        ssh_scp_get(host, username, password, None, file_path + filename, "./")
                        except:
                                print("{} cannot be found...".format(filename))

class download():

    @staticmethod
    def reporthook(count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                        (percent, progress_size / (1024 * 1024), speed, duration))
        sys.stdout.flush()

    @staticmethod
    def glade(url="http://glade.elte.hu/GLADE_2.3.txt", local_path=None):
        print("Downloading GLADE galaxy catalog ...")
        if local_path==None:
            local_path = pkg_resources.resource_filename('gTranRec', 'data')
            if not os.path.exists(local_path):
                os.makedirs(local_path)

        out_txt = os.path.join(local_path,'GLADE.txt')
        urlretrieve(url, out_txt, download.reporthook)

