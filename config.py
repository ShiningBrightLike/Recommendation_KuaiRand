"""Single source of truth for paths, feature schema, and training defaults.

All pipeline scripts import from this module instead of re-declaring feature
lists or hyperparameters, so schema/config changes happen in exactly one place.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to the repository root)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
KUAI_DIR = ROOT / "KuaiRand-Pure"
DATA_DIR = KUAI_DIR / "data"
PROCESSED_DIR = KUAI_DIR / "data_processed"
SAVED_DIR = KUAI_DIR / "saved"
RUNS_DIR = SAVED_DIR / "runs"

TRAIN_LOG_FILE = DATA_DIR / "log_standard_4_08_to_4_21_pure.csv"
TEST_LOG_FILE = DATA_DIR / "log_standard_4_22_to_5_08_pure.csv"
USER_FEATURES_FILE = DATA_DIR / "user_features_pure.csv"
VIDEO_FEATURES_BASIC_FILE = DATA_DIR / "video_features_basic_pure.csv"
VIDEO_FEATURES_STATISTIC_FILE = DATA_DIR / "video_features_statistic_pure.csv"

# ---------------------------------------------------------------------------
# Time-based split
# ---------------------------------------------------------------------------
# Rows with date >= VAL_START_DATE (inclusive) inside the training log become
# the validation split. The test log (4/22-5/08) is only used for final
# evaluation after training/early-stopping has finished.
VAL_START_DATE = "20220416"

# ---------------------------------------------------------------------------
# Feature schema (keep this list as the single source of truth)
# ---------------------------------------------------------------------------
CATEGORICAL_COLS = [
    "date", "hourmin", "tab",
    "user_active_degree", "is_lowactive_period", "is_live_streamer", "is_video_author",
    "onehot_feat0", "onehot_feat1", "onehot_feat2", "onehot_feat3", "onehot_feat4",
    "onehot_feat5", "onehot_feat6", "onehot_feat7", "onehot_feat8", "onehot_feat9",
    "onehot_feat10", "onehot_feat11", "onehot_feat12", "onehot_feat13", "onehot_feat14",
    "onehot_feat15", "onehot_feat16", "onehot_feat17",
    "follow_user_num_range", "fans_user_num_range", "friend_user_num_range", "register_days_range",
    "video_type", "upload_dt", "upload_type", "visible_status", "music_type", "tag",
]

NUMERIC_COLS = [
    "follow_user_num", "fans_user_num", "friend_user_num", "register_days",
    "video_duration", "server_width", "server_height",
    "counts", "show_cnt", "show_user_num", "play_cnt",
    "play_user_num", "play_duration", "complete_play_cnt",
    "complete_play_user_num", "valid_play_cnt", "valid_play_user_num",
    "long_time_play_cnt", "long_time_play_user_num", "short_time_play_cnt",
    "short_time_play_user_num", "play_progress", "comment_stay_duration",
    "like_cnt", "like_user_num", "click_like_cnt", "double_click_cnt",
    "cancel_like_cnt", "cancel_like_user_num", "comment_cnt",
    "comment_user_num", "direct_comment_cnt", "reply_comment_cnt",
    "delete_comment_cnt", "delete_comment_user_num", "comment_like_cnt",
    "comment_like_user_num", "follow_cnt", "follow_user_num1",
    "cancel_follow_cnt", "cancel_follow_user_num", "share_cnt",
    "share_user_num", "download_cnt", "download_user_num", "report_cnt",
    "report_user_num", "reduce_similar_cnt", "reduce_similar_user_num",
    "collect_cnt", "collect_user_num", "cancel_collect_cnt",
    "cancel_collect_user_num", "direct_comment_user_num",
    "reply_comment_user_num", "share_all_cnt", "share_all_user_num",
    "outsite_share_all_cnt",
]

LABEL_COLS = ["is_click", "is_like", "is_follow", "is_comment"]

# Mapping from processed split name to (X, y) file names under PROCESSED_DIR.
SPLIT_FILES = {
    "train": ("processed_X.parquet", "processed_y.parquet"),
    "val": ("processed_X_val.parquet", "processed_y_val.parquet"),
    "test": ("processed_X_test.parquet", "processed_y_test.parquet"),
}

# ---------------------------------------------------------------------------
# Model architecture defaults
# ---------------------------------------------------------------------------
EMBED_DIM = 8
NUM_EXPERTS = 8
EXPERT_UNITS = 64
TOWER_UNITS = 32

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
RANDOM_SEED = 2025
EPOCHS = 30
BATCH_SIZE = 1024
EARLY_STOP_PATIENCE = 5
LEARNING_RATE = 1e-3

# Order must match LABEL_COLS (output_i corresponds to LABEL_COLS[i]).
LOSS_WEIGHTS = {
    "is_click": 1.0,
    "is_like": 1.0,
    "is_follow": 0.5,
    "is_comment": 0.1,
}
