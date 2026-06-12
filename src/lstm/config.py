# Model Architecture Parameters
WINDOW_SIZE = 5
BATCH_SIZE = 512
EPOCHS = 50
TRAIN_RATIO = 0.8

# Data Paths and Output Files
DATA_DIR = r"C:\Users\DUY\Downloads\CSV"
MODEL_TIER1_PATH = "IDS_Tier1_Binary.keras"
MODEL_TIER2_PATH = "IDS_Tier2_Multiclass.keras"
OUTPUT_REPORT_PATH = "Ket_Qua_Du_Doan_IDS.csv"

# Original UNSW-NB15 Raw Dataset Columns
UNSW_COLUMNS = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur',
    'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service',
    'sload', 'dload', 'spkts', 'dpkts', 'swin', 'dwin', 'stcpb',
    'dtcpb', 'smean', 'dmean', 'trans_depth', 'res_bdy_len', 'sjit',
    'djit', 'stime', 'ltime', 'sintpkt', 'dintpkt', 'tcprtt',
    'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl',
    'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src',
    'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm',
    'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label'
]
