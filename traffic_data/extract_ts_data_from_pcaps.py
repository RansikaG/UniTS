from scapy.all import rdpcap
import os
from tqdm import tqdm

# Base directory where class folders are located
base_dir = "/home/ransika/cstnet/cstnet-tls-1.3"
output_file = "cstnet.ts"

def write_ts_header(file, class_labels):
    """
    Write the correct header format for sktime TSFile format.
    """
    # Join class labels with spaces as per TSFile format requirement
    class_label_str = ' '.join(class_labels)

    header = f"""@problemName cstnet
@timeStamps false
@missing false
@univariate false
@dimensions 3
@equalLength false
@seriesLength -1
@classLabel true {class_label_str}
@data
"""
    file.write(header)

# Function to process a single pcap file and extract session data
def process_pcap(pcap_file_path, class_label):
    packets = rdpcap(pcap_file_path)
    timestamps = []
    packet_sizes = []
    directions = []

    # Get the source IP of the first packet
    first_packet = packets[0]
    src_ip = first_packet[0][1].src if first_packet.haslayer('IP') else None

    if src_ip:
        for i,packet in enumerate(packets):
            previous_time = 0
            if packet.haslayer('IP'):
                if i == 0 :
                    timestamps.append(float(previous_time))
                else:
                    timestamps.append(float(packet.time)-previous_time)  # Packet timestamp
                packet_sizes.append(len(packet))  # Packet size
                direction = 1 if packet[1].src == src_ip else 0  # 1 for outgoing, 0 for incoming
                directions.append(direction)
                previous_time =float(packet.time)

    return timestamps, packet_sizes, directions, class_label

# Open output .ts file and write the header
# Get the list of class folders for labels
class_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))][12:]
class_labels = [folder for folder in class_folders]  # Extracting class labels from folder names

with open(output_file, "w") as ts_file:
    write_ts_header(ts_file, class_labels)

    # Set up the progress bar
    total_pcaps = sum([len(os.listdir(os.path.join(base_dir, f))) for f in class_folders])
    with tqdm(total=total_pcaps, desc="Processing PCAP files") as pbar:

        # Iterate through class folders and pcaps
        for class_folder in class_folders:
            class_folder_path = os.path.join(base_dir, class_folder)
            class_label = class_folder  # Assign folder name as class label

            for pcap_file in os.listdir(class_folder_path):
                if pcap_file.endswith(".pcap"):
                    pcap_path = os.path.join(class_folder_path, pcap_file)

                    # Process the pcap file and extract session data
                    timestamps, packet_sizes, directions, label = process_pcap(pcap_path, class_label)

                    if timestamps and packet_sizes and directions:
                        # Format the session data as multivariate time series
                        formatted_session = f"{','.join(map(str, timestamps))}:{','.join(map(str, packet_sizes))}:{','.join(map(str, directions))}:{label}\n"

                        # Write the session data to the .ts file
                        ts_file.write(formatted_session)

                    # Update the progress bar
                    pbar.update(1)