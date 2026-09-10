# 置换重要度报告

- model: `KuaiRand-Pure\saved\runs\fi-shadow_20260910_010531\model.keras`
- split: `val` / rows: `190802`
- gate tasks: is_click, is_like | cutoff: 0.001
- baseline AUC: {"is_click": 0.7377557171711931, "is_like": 0.8317722430087254, "is_follow": 0.8274558926941603, "is_comment": 0.7614221941294272}

## 结论（按总体重要度排序）

| 特征 | 类型 | 影子 | 总体(均值±std) | 判定 | is_click | is_like | is_follow | is_comment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tab | categorical | False | 0.06120 ± 0.00119 | 通过 | 0.09259 ± 0.00133 | 0.02981 ± 0.00105 | 0.00533 ± 0.00035 | 0.02890 ± 0.00278 |
| onehot_feat3 | categorical | False | 0.03162 ± 0.00035 | 通过 | 0.01946 ± 0.00034 | 0.04377 ± 0.00036 | 0.00516 ± 0.00374 | 0.00136 ± 0.00318 |
| valid_play_user_num | numeric | False | 0.01991 ± 0.00074 | 通过 | 0.03562 ± 0.00065 | 0.00421 ± 0.00083 | 0.00547 ± 0.00227 | 0.00269 ± 0.00226 |
| valid_play_cnt | numeric | False | 0.01902 ± 0.00028 | 通过 | 0.03623 ± 0.00015 | 0.00181 ± 0.00042 | 0.00437 ± 0.00219 | 0.00499 ± 0.00348 |
| onehot_feat8 | categorical | False | 0.01532 ± 0.00052 | 通过 | 0.00820 ± 0.00025 | 0.02244 ± 0.00080 | 0.00722 ± 0.00235 | 0.00488 ± 0.00147 |
| follow_user_num | numeric | False | 0.01416 ± 0.00049 | 通过 | 0.00208 ± 0.00026 | 0.02624 ± 0.00072 | 0.08302 ± 0.00559 | 0.00583 ± 0.00100 |
| short_time_play_user_num | numeric | False | 0.01312 ± 0.00075 | 通过 | 0.02354 ± 0.00029 | 0.00270 ± 0.00121 | 0.00891 ± 0.00515 | 0.00713 ± 0.00194 |
| double_click_cnt | numeric | False | 0.01294 ± 0.00093 | 通过 | 0.00211 ± 0.00013 | 0.02376 ± 0.00173 | 0.00088 ± 0.00070 | 0.00369 ± 0.00215 |
| short_time_play_cnt | numeric | False | 0.01204 ± 0.00051 | 通过 | 0.02015 ± 0.00067 | 0.00394 ± 0.00035 | 0.00195 ± 0.00489 | 0.00488 ± 0.00068 |
| onehot_feat1 | categorical | False | 0.01104 ± 0.00048 | 通过 | 0.00589 ± 0.00020 | 0.01620 ± 0.00076 | 0.00234 ± 0.00227 | 0.02028 ± 0.00204 |
| follow_user_num_range | categorical | False | 0.00811 ± 0.00024 | 通过 | 0.00285 ± 0.00011 | 0.01338 ± 0.00037 | 0.02981 ± 0.00209 | 0.00161 ± 0.00112 |
| play_progress | numeric | False | 0.00717 ± 0.00036 | 通过 | 0.00577 ± 0.00034 | 0.00858 ± 0.00037 | 0.00913 ± 0.00374 | 0.00299 ± 0.00195 |
| like_user_num | numeric | False | 0.00652 ± 0.00031 | 通过 | 0.00044 ± 0.00006 | 0.01260 ± 0.00055 | -0.00025 ± 0.00098 | 0.00333 ± 0.00139 |
| like_cnt | numeric | False | 0.00590 ± 0.00025 | 通过 | 0.00074 ± 0.00014 | 0.01106 ± 0.00036 | 0.00024 ± 0.00169 | 0.00171 ± 0.00184 |
| play_user_num | numeric | False | 0.00576 ± 0.00081 | 通过 | 0.00525 ± 0.00023 | 0.00627 ± 0.00139 | 0.00447 ± 0.00105 | 0.00645 ± 0.00242 |
| user_active_degree | categorical | False | 0.00564 ± 0.00033 | 通过 | 0.00543 ± 0.00013 | 0.00586 ± 0.00053 | 0.00896 ± 0.00101 | 0.00408 ± 0.00015 |
| long_time_play_cnt | numeric | False | 0.00495 ± 0.00019 | 通过 | 0.00909 ± 0.00017 | 0.00080 ± 0.00022 | 0.00035 ± 0.00138 | 0.00016 ± 0.00014 |
| long_time_play_user_num | numeric | False | 0.00461 ± 0.00022 | 通过 | 0.00771 ± 0.00015 | 0.00151 ± 0.00029 | 0.00447 ± 0.00101 | 0.00164 ± 0.00089 |
| show_user_num | numeric | False | 0.00441 ± 0.00017 | 通过 | 0.00228 ± 0.00004 | 0.00654 ± 0.00031 | 0.00622 ± 0.00188 | 0.00395 ± 0.00508 |
| friend_user_num | numeric | False | 0.00385 ± 0.00062 | 通过 | 0.00122 ± 0.00003 | 0.00648 ± 0.00121 | 0.01309 ± 0.00604 | 0.00084 ± 0.00048 |
| video_duration | numeric | False | 0.00333 ± 0.00017 | 通过 | 0.00318 ± 0.00016 | 0.00347 ± 0.00018 | 0.00196 ± 0.00223 | 0.00125 ± 0.00115 |
| click_like_cnt | numeric | False | 0.00319 ± 0.00020 | 通过 | 0.00064 ± 0.00003 | 0.00574 ± 0.00037 | 0.00114 ± 0.00069 | 0.00471 ± 0.00090 |
| friend_user_num_range | categorical | False | 0.00318 ± 0.00034 | 通过 | 0.00167 ± 0.00018 | 0.00468 ± 0.00051 | 0.00848 ± 0.00155 | 0.00215 ± 0.00255 |
| onehot_feat7 | categorical | False | 0.00308 ± 0.00037 | 通过 | 0.00236 ± 0.00018 | 0.00380 ± 0.00055 | 0.00119 ± 0.00027 | 0.00352 ± 0.00063 |
| tag | categorical | False | 0.00308 ± 0.00021 | 通过 | 0.00393 ± 0.00015 | 0.00222 ± 0.00027 | 0.00186 ± 0.00161 | 0.00072 ± 0.00141 |
| reduce_similar_user_num | numeric | False | 0.00284 ± 0.00023 | 通过 | 0.00146 ± 0.00005 | 0.00421 ± 0.00042 | 0.00016 ± 0.00044 | 0.00122 ± 0.00024 |
| onehot_feat0 | categorical | False | 0.00274 ± 0.00032 | 通过 | 0.00297 ± 0.00003 | 0.00251 ± 0.00060 | 0.00098 ± 0.00218 | 0.00959 ± 0.00157 |
| register_days_range | categorical | False | 0.00274 ± 0.00012 | 通过 | 0.00104 ± 0.00016 | 0.00443 ± 0.00007 | 0.00567 ± 0.00123 | 0.00277 ± 0.00148 |
| cancel_like_user_num | numeric | False | 0.00266 ± 0.00018 | 通过 | 0.00400 ± 0.00009 | 0.00132 ± 0.00027 | 0.00065 ± 0.00055 | 0.00266 ± 0.00022 |
| fans_user_num_range | categorical | False | 0.00261 ± 0.00051 | 通过 | 0.00129 ± 0.00021 | 0.00393 ± 0.00081 | 0.00058 ± 0.00227 | 0.00261 ± 0.00168 |
| register_days | numeric | False | 0.00257 ± 0.00024 | 通过 | 0.00202 ± 0.00020 | 0.00311 ± 0.00028 | 0.00096 ± 0.00046 | 0.00053 ± 0.00098 |
| play_cnt | numeric | False | 0.00221 ± 0.00031 | 通过 | 0.00325 ± 0.00010 | 0.00116 ± 0.00051 | 0.00175 ± 0.00196 | 0.00153 ± 0.00097 |
| onehot_feat4 | categorical | False | 0.00219 ± 0.00020 | 通过 | 0.00146 ± 0.00015 | 0.00291 ± 0.00024 | 0.00004 ± 0.00135 | 0.00079 ± 0.00102 |
| onehot_feat2 | categorical | False | 0.00213 ± 0.00024 | 通过 | 0.00215 ± 0.00024 | 0.00211 ± 0.00024 | 0.00305 ± 0.00198 | 0.00914 ± 0.00607 |
| play_duration | numeric | False | 0.00202 ± 0.00017 | 通过 | 0.00251 ± 0.00007 | 0.00154 ± 0.00027 | 0.00246 ± 0.00333 | 0.00201 ± 0.00124 |
| collect_user_num | numeric | False | 0.00198 ± 0.00022 | 通过 | 0.00136 ± 0.00013 | 0.00259 ± 0.00030 | -0.00014 ± 0.00058 | -0.00006 ± 0.00099 |
| is_live_streamer | categorical | False | 0.00189 ± 0.00037 | 通过 | 0.00139 ± 0.00005 | 0.00238 ± 0.00070 | -0.00139 ± 0.00072 | 0.00110 ± 0.00141 |
| outsite_share_all_cnt | numeric | False | 0.00185 ± 0.00013 | 通过 | 0.00244 ± 0.00006 | 0.00126 ± 0.00020 | 0.00257 ± 0.00069 | 0.00120 ± 0.00166 |
| onehot_feat10 | categorical | False | 0.00176 ± 0.00010 | 通过 | 0.00118 ± 0.00010 | 0.00235 ± 0.00010 | 0.00308 ± 0.00044 | 0.00024 ± 0.00229 |
| counts | numeric | False | 0.00173 ± 0.00021 | 通过 | 0.00096 ± 0.00007 | 0.00251 ± 0.00035 | 0.00088 ± 0.00140 | 0.00136 ± 0.00060 |
| cancel_like_cnt | numeric | False | 0.00156 ± 0.00003 | 通过 | 0.00139 ± 0.00005 | 0.00173 ± 0.00002 | 0.00349 ± 0.00290 | 0.00532 ± 0.00082 |
| cancel_collect_user_num | numeric | False | 0.00152 ± 0.00032 | 通过 | 0.00139 ± 0.00012 | 0.00165 ± 0.00053 | 0.00183 ± 0.00107 | 0.00155 ± 0.00093 |
| onehot_feat9 | categorical | False | 0.00150 ± 0.00027 | 通过 | 0.00105 ± 0.00005 | 0.00195 ± 0.00049 | 0.00234 ± 0.00222 | 0.00102 ± 0.00082 |
| collect_cnt | numeric | False | 0.00139 ± 0.00025 | 通过 | 0.00032 ± 0.00008 | 0.00247 ± 0.00042 | 0.00115 ± 0.00150 | 0.00161 ± 0.00051 |
| onehot_feat11 | categorical | False | 0.00139 ± 0.00017 | 通过 | 0.00192 ± 0.00012 | 0.00086 ± 0.00023 | 0.00220 ± 0.00091 | 0.02625 ± 0.00237 |
| reply_comment_cnt | numeric | False | 0.00137 ± 0.00015 | 通过 | 0.00203 ± 0.00010 | 0.00071 ± 0.00019 | 0.00070 ± 0.00268 | 0.00598 ± 0.00059 |
| onehot_feat12 | categorical | False | 0.00137 ± 0.00026 | 通过 | 0.00150 ± 0.00008 | 0.00124 ± 0.00044 | 0.00107 ± 0.00252 | 0.00288 ± 0.00068 |
| onehot_feat6 | categorical | False | 0.00136 ± 0.00020 | 通过 | 0.00158 ± 0.00013 | 0.00113 ± 0.00027 | 0.00186 ± 0.00035 | 0.00364 ± 0.00145 |
| share_user_num | numeric | False | 0.00134 ± 0.00014 | 通过 | 0.00086 ± 0.00012 | 0.00182 ± 0.00017 | -0.00122 ± 0.00068 | 0.00474 ± 0.00297 |
| show_cnt | numeric | False | 0.00126 ± 0.00003 | 通过 | 0.00093 ± 0.00002 | 0.00158 ± 0.00003 | 0.00098 ± 0.00124 | -0.00038 ± 0.00083 |
| upload_type | categorical | False | 0.00098 ± 0.00020 | 不通过 | 0.00067 ± 0.00003 | 0.00130 ± 0.00037 | -0.00295 ± 0.00068 | 0.00036 ± 0.00012 |
| hourmin | categorical | False | 0.00095 ± 0.00009 | 不通过 | 0.00130 ± 0.00008 | 0.00059 ± 0.00010 | 0.00063 ± 0.00112 | 0.00090 ± 0.00132 |
| cancel_follow_cnt | numeric | False | 0.00094 ± 0.00035 | 不通过 | 0.00035 ± 0.00003 | 0.00153 ± 0.00068 | 0.00179 ± 0.00099 | -0.00055 ± 0.00097 |
| reduce_similar_cnt | numeric | False | 0.00092 ± 0.00027 | 不通过 | 0.00055 ± 0.00009 | 0.00128 ± 0.00045 | -0.00069 ± 0.00178 | 0.00214 ± 0.00006 |
| comment_like_user_num | numeric | False | 0.00087 ± 0.00029 | 不通过 | 0.00092 ± 0.00003 | 0.00082 ± 0.00054 | -0.00003 ± 0.00134 | 0.00216 ± 0.00066 |
| cancel_collect_cnt | numeric | False | 0.00086 ± 0.00002 | 不通过 | 0.00116 ± 0.00001 | 0.00057 ± 0.00003 | 0.00168 ± 0.00008 | 0.00143 ± 0.00159 |
| share_all_user_num | numeric | False | 0.00084 ± 0.00007 | 不通过 | 0.00072 ± 0.00006 | 0.00096 ± 0.00009 | -0.00112 ± 0.00158 | 0.00053 ± 0.00079 |
| share_cnt | numeric | False | 0.00084 ± 0.00025 | 不通过 | 0.00061 ± 0.00008 | 0.00106 ± 0.00042 | -0.00078 ± 0.00108 | -0.00038 ± 0.00144 |
| is_video_author | categorical | False | 0.00079 ± 0.00007 | 不通过 | 0.00064 ± 0.00006 | 0.00093 ± 0.00007 | 0.00244 ± 0.00114 | -0.00024 ± 0.00068 |
| follow_user_num1 | numeric | False | 0.00079 ± 0.00011 | 不通过 | 0.00111 ± 0.00016 | 0.00046 ± 0.00007 | 0.01144 ± 0.00288 | 0.00104 ± 0.00143 |
| follow_cnt | numeric | False | 0.00077 ± 0.00012 | 不通过 | 0.00079 ± 0.00005 | 0.00075 ± 0.00018 | 0.01700 ± 0.00123 | 0.00033 ± 0.00160 |
| server_width | numeric | False | 0.00074 ± 0.00023 | 不通过 | 0.00050 ± 0.00013 | 0.00099 ± 0.00033 | -0.00100 ± 0.00093 | 0.00051 ± 0.00217 |
| comment_cnt | numeric | False | 0.00074 ± 0.00011 | 不通过 | 0.00086 ± 0.00005 | 0.00062 ± 0.00018 | 0.00121 ± 0.00101 | 0.00996 ± 0.00197 |
| comment_like_cnt | numeric | False | 0.00065 ± 0.00015 | 不通过 | 0.00059 ± 0.00003 | 0.00072 ± 0.00026 | -0.00170 ± 0.00110 | 0.00427 ± 0.00075 |
| complete_play_cnt | numeric | False | 0.00064 ± 0.00021 | 不通过 | 0.00045 ± 0.00010 | 0.00083 ± 0.00032 | 0.00150 ± 0.00050 | 0.00119 ± 0.00014 |
| complete_play_user_num | numeric | False | 0.00062 ± 0.00014 | 不通过 | 0.00071 ± 0.00020 | 0.00053 ± 0.00009 | -0.00203 ± 0.00121 | -0.00040 ± 0.00013 |
| download_cnt | numeric | False | 0.00058 ± 0.00010 | 不通过 | 0.00016 ± 0.00005 | 0.00099 ± 0.00015 | -0.00125 ± 0.00081 | 0.00026 ± 0.00026 |
| download_user_num | numeric | False | 0.00055 ± 0.00012 | 不通过 | 0.00020 ± 0.00011 | 0.00090 ± 0.00012 | -0.00084 ± 0.00018 | 0.00326 ± 0.00177 |
| reply_comment_user_num | numeric | False | 0.00054 ± 0.00008 | 不通过 | 0.00026 ± 0.00001 | 0.00082 ± 0.00015 | -0.00103 ± 0.00186 | 0.00080 ± 0.00122 |
| server_height | numeric | False | 0.00054 ± 0.00005 | 不通过 | 0.00037 ± 0.00003 | 0.00071 ± 0.00006 | -0.00179 ± 0.00059 | -0.00109 ± 0.00110 |
| onehot_feat16 | categorical | False | 0.00051 ± 0.00009 | 不通过 | 0.00027 ± 0.00005 | 0.00075 ± 0.00014 | 0.00166 ± 0.00038 | 0.00167 ± 0.00029 |
| date | categorical | False | 0.00051 ± 0.00016 | 不通过 | 0.00050 ± 0.00021 | 0.00052 ± 0.00010 | 0.00335 ± 0.00236 | 0.00068 ± 0.00174 |
| comment_user_num | numeric | False | 0.00049 ± 0.00022 | 不通过 | 0.00041 ± 0.00003 | 0.00042 ± 0.00063 | 0.00054 ± 0.00152 | 0.00660 ± 0.00119 |
| delete_comment_user_num | numeric | False | 0.00048 ± 0.00018 | 不通过 | 0.00012 ± 0.00009 | 0.00085 ± 0.00027 | 0.00053 ± 0.00127 | 0.00354 ± 0.00102 |
| direct_comment_user_num | numeric | False | 0.00047 ± 0.00022 | 不通过 | 0.00035 ± 0.00011 | 0.00060 ± 0.00033 | 0.00304 ± 0.00404 | 0.00099 ± 0.00192 |
| comment_stay_duration | numeric | False | 0.00046 ± 0.00010 | 不通过 | 0.00081 ± 0.00013 | -0.00006 ± 0.00015 | 0.00335 ± 0.00232 | 0.02020 ± 0.00074 |
| onehot_feat14 | categorical | False | 0.00046 ± 0.00015 | 不通过 | 0.00066 ± 0.00013 | 0.00025 ± 0.00018 | 0.00024 ± 0.00011 | 0.00149 ± 0.00051 |
| share_all_cnt | numeric | False | 0.00041 ± 0.00012 | 不通过 | 0.00038 ± 0.00003 | 0.00043 ± 0.00021 | 0.00131 ± 0.00029 | 0.00331 ± 0.00062 |
| report_user_num | numeric | False | 0.00035 ± 0.00010 | 不通过 | 0.00051 ± 0.00011 | 0.00018 ± 0.00009 | 0.00024 ± 0.00053 | -0.00012 ± 0.00069 |
| cancel_follow_user_num | numeric | False | 0.00031 ± 0.00007 | 不通过 | 0.00043 ± 0.00010 | 0.00019 ± 0.00004 | 0.00206 ± 0.00022 | 0.00037 ± 0.00101 |
| direct_comment_cnt | numeric | False | 0.00029 ± 0.00008 | 不通过 | 0.00038 ± 0.00002 | 0.00021 ± 0.00015 | 0.00324 ± 0.00142 | 0.00688 ± 0.00178 |
| upload_dt | categorical | False | 0.00029 ± 0.00016 | 不通过 | 0.00014 ± 0.00001 | 0.00037 ± 0.00042 | 0.00009 ± 0.00168 | 0.00010 ± 0.00057 |
| delete_comment_cnt | numeric | False | 0.00027 ± 0.00020 | 不通过 | 0.00016 ± 0.00005 | 0.00039 ± 0.00035 | 0.00139 ± 0.00022 | 0.00021 ± 0.00017 |
| fans_user_num | numeric | False | 0.00027 ± 0.00006 | 不通过 | 0.00010 ± 0.00001 | 0.00044 ± 0.00012 | 0.00130 ± 0.00215 | 0.00070 ± 0.00145 |
| report_cnt | numeric | False | 0.00024 ± 0.00008 | 不通过 | 0.00041 ± 0.00013 | -0.00001 ± 0.00008 | 0.00102 ± 0.00092 | 0.00044 ± 0.00016 |
| onehot_feat15 | categorical | False | 0.00020 ± 0.00006 | 不通过 | 0.00028 ± 0.00004 | 0.00008 ± 0.00014 | -0.00063 ± 0.00005 | 0.00009 ± 0.00028 |
| onehot_feat13 | categorical | False | 0.00016 ± 0.00012 | 不通过 | 0.00013 ± 0.00002 | 0.00020 ± 0.00022 | 0.00015 ± 0.00105 | 0.00108 ± 0.00006 |
| music_type | categorical | False | 0.00013 ± 0.00003 | 不通过 | 0.00008 ± 0.00003 | -0.00019 ± 0.00003 | -0.00061 ± 0.00008 | 0.00017 ± 0.00032 |
| shadow_0 | numeric | True | 0.00010 ± 0.00006 | 不通过 | 0.00007 ± 0.00006 | -0.00007 ± 0.00013 | 0.00035 ± 0.00050 | 0.00020 ± 0.00088 |
| onehot_feat17 | categorical | False | 0.00006 ± 0.00004 | 不通过 | 0.00006 ± 0.00001 | 0.00007 ± 0.00006 | -0.00036 ± 0.00053 | 0.00085 ± 0.00066 |
| video_type | categorical | False | 0.00003 ± 0.00002 | 不通过 | -0.00000 ± 0.00002 | -0.00004 ± 0.00002 | -0.00012 ± 0.00034 | -0.00022 ± 0.00016 |
| onehot_feat5 | categorical | False | 0.00002 ± 0.00001 | 不通过 | 0.00001 ± 0.00000 | -0.00000 ± 0.00005 | -0.00003 ± 0.00003 | -0.00001 ± 0.00013 |
| is_lowactive_period | categorical | False | 0.00000 ± 0.00000 | 不通过 | 0.00000 ± 0.00000 | 0.00000 ± 0.00000 | 0.00000 ± 0.00000 | 0.00000 ± 0.00000 |
| visible_status | categorical | False | 0.00000 ± 0.00000 | 不通过 | 0.00000 ± 0.00000 | 0.00000 ± 0.00000 | 0.00000 ± 0.00000 | 0.00000 ± 0.00000 |

## 噪声参考

{
  "present": true,
  "features": [
    "shadow_0"
  ],
  "overall_mean": 9.506367714886185e-05,
  "overall_std": 0.0
}

## 相关特征提示（置换重要度会被摊薄，请合并解读）

| 特征 A | 特征 B | Pearson r |
| --- | --- | --- |
| show_cnt | show_user_num | 0.9962 |
| show_cnt | play_cnt | 0.9867 |
| show_cnt | play_user_num | 0.983 |
| show_cnt | valid_play_cnt | 0.9711 |
| show_cnt | valid_play_user_num | 0.9668 |
| show_cnt | long_time_play_cnt | 0.9597 |
| show_cnt | long_time_play_user_num | 0.956 |
| show_cnt | short_time_play_cnt | 0.9629 |
| show_cnt | short_time_play_user_num | 0.9587 |
| show_cnt | reduce_similar_cnt | 0.9056 |
| show_user_num | play_cnt | 0.9896 |
| show_user_num | play_user_num | 0.9902 |
| show_user_num | valid_play_cnt | 0.9764 |
| show_user_num | valid_play_user_num | 0.9762 |
| show_user_num | long_time_play_cnt | 0.964 |
| show_user_num | long_time_play_user_num | 0.9645 |
| show_user_num | short_time_play_cnt | 0.9638 |
| show_user_num | short_time_play_user_num | 0.9625 |
| show_user_num | reduce_similar_cnt | 0.9014 |
| play_cnt | play_user_num | 0.9984 |
| play_cnt | valid_play_cnt | 0.9905 |
| play_cnt | valid_play_user_num | 0.9872 |
| play_cnt | long_time_play_cnt | 0.9819 |
| play_cnt | long_time_play_user_num | 0.9791 |
| play_cnt | short_time_play_cnt | 0.9674 |
| play_cnt | short_time_play_user_num | 0.9638 |
| play_user_num | valid_play_cnt | 0.9902 |
| play_user_num | valid_play_user_num | 0.99 |
| play_user_num | long_time_play_cnt | 0.9817 |
| play_user_num | long_time_play_user_num | 0.9818 |
| play_user_num | short_time_play_cnt | 0.9656 |
| play_user_num | short_time_play_user_num | 0.9642 |
| complete_play_cnt | complete_play_user_num | 0.9992 |
| complete_play_cnt | long_time_play_cnt | 0.9019 |
| complete_play_cnt | long_time_play_user_num | 0.9021 |
| complete_play_user_num | long_time_play_cnt | 0.9002 |
| complete_play_user_num | long_time_play_user_num | 0.9023 |
| valid_play_cnt | valid_play_user_num | 0.9984 |
| valid_play_cnt | long_time_play_cnt | 0.9953 |
| valid_play_cnt | long_time_play_user_num | 0.9943 |
| valid_play_cnt | short_time_play_cnt | 0.9278 |
| valid_play_cnt | short_time_play_user_num | 0.9236 |
| valid_play_user_num | long_time_play_cnt | 0.9932 |
| valid_play_user_num | long_time_play_user_num | 0.9953 |
| valid_play_user_num | short_time_play_cnt | 0.9232 |
| valid_play_user_num | short_time_play_user_num | 0.9209 |
| long_time_play_cnt | long_time_play_user_num | 0.9985 |
| long_time_play_cnt | short_time_play_cnt | 0.9174 |
| long_time_play_cnt | short_time_play_user_num | 0.9125 |
| long_time_play_user_num | short_time_play_cnt | 0.913 |
| long_time_play_user_num | short_time_play_user_num | 0.9101 |
| short_time_play_cnt | short_time_play_user_num | 0.9992 |
| like_cnt | like_user_num | 0.9999 |
| like_cnt | click_like_cnt | 0.9887 |
| like_cnt | double_click_cnt | 0.9927 |
| like_user_num | click_like_cnt | 0.9874 |
| like_user_num | double_click_cnt | 0.9937 |
| click_like_cnt | double_click_cnt | 0.9635 |
| cancel_like_cnt | cancel_like_user_num | 0.9944 |
| cancel_like_cnt | cancel_collect_cnt | 0.9062 |
| cancel_like_user_num | cancel_collect_cnt | 0.9251 |
| cancel_like_user_num | cancel_collect_user_num | 0.917 |
| comment_cnt | comment_user_num | 0.9905 |
| comment_cnt | direct_comment_cnt | 0.9907 |
| comment_cnt | reply_comment_cnt | 0.9828 |
| comment_cnt | direct_comment_user_num | 0.9865 |
| comment_cnt | reply_comment_user_num | 0.979 |
| comment_user_num | direct_comment_cnt | 0.9729 |
| comment_user_num | reply_comment_cnt | 0.9848 |
| comment_user_num | direct_comment_user_num | 0.9958 |
| comment_user_num | reply_comment_user_num | 0.9871 |
| direct_comment_cnt | reply_comment_cnt | 0.9485 |
| direct_comment_cnt | direct_comment_user_num | 0.9784 |
| direct_comment_cnt | reply_comment_user_num | 0.9445 |
| reply_comment_cnt | direct_comment_user_num | 0.968 |
| reply_comment_cnt | reply_comment_user_num | 0.9967 |
| comment_like_cnt | comment_like_user_num | 0.979 |
| follow_cnt | follow_user_num1 | 1.0 |
| cancel_follow_cnt | cancel_follow_user_num | 0.9999 |
| share_cnt | share_user_num | 0.9987 |
| share_cnt | share_all_cnt | 0.9878 |
| share_cnt | share_all_user_num | 0.9867 |
| share_user_num | share_all_cnt | 0.9873 |
| share_user_num | share_all_user_num | 0.9878 |
| download_cnt | download_user_num | 0.996 |
| report_cnt | report_user_num | 0.9762 |
| reduce_similar_cnt | reduce_similar_user_num | 0.9977 |
| collect_cnt | collect_user_num | 0.9998 |
| cancel_collect_cnt | cancel_collect_user_num | 0.9993 |
| direct_comment_user_num | reply_comment_user_num | 0.9686 |
| share_all_cnt | share_all_user_num | 0.9993 |
| share_all_cnt | outsite_share_all_cnt | 0.9161 |
| share_all_user_num | outsite_share_all_cnt | 0.9079 |
