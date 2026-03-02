log_file_path = "data/raw/HDFS_v1/HDFS.log"
anomaly_label_path = "data/raw/HDFS_v1/preprocessed/anomaly_label.csv"





with open(anomaly_label_path, 'r') as file:
    for i, line in enumerate(file):
        if i < 20:
            print(line.rstrip('\n'))
        else:
            break