# 置换重要度报告

- model: `KuaiRand-Pure\saved\runs\fi-v2-base_20260915_090053\model.keras`
- split: `val` / rows: `190802`
- gate tasks: is_click, is_like | cutoff: 0.001
- baseline AUC: {"is_click": 0.7352329927741181, "is_like": 0.8271236490817867, "is_follow": 0.8108411900309819, "is_comment": 0.7440773351109333}

## 结论（按总体重要度排序）

| 特征 | 类型 | 影子 | 总体(均值±std) | 判定 | is_click | is_like | is_follow | is_comment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tab | categorical | False | 0.06315 ± 0.00151 | 通过 | 0.09400 ± 0.00168 | 0.03230 ± 0.00135 | 0.00325 ± 0.00341 | 0.03935 ± 0.00377 |
| onehot_feat3 | categorical | False | 0.03475 ± 0.00097 | 通过 | 0.02089 ± 0.00071 | 0.04861 ± 0.00124 | 0.00615 ± 0.00747 | 0.00545 ± 0.00633 |
| valid_play_cnt | numeric | False | 0.02523 ± 0.00038 | 通过 | 0.04007 ± 0.00015 | 0.01039 ± 0.00060 | 0.00760 ± 0.00456 | 0.00563 ± 0.00085 |
| valid_play_user_num | numeric | False | 0.02290 ± 0.00114 | 通过 | 0.03780 ± 0.00067 | 0.00800 ± 0.00161 | 0.00797 ± 0.00335 | 0.00224 ± 0.00393 |
| onehot_feat8 | categorical | False | 0.01745 ± 0.00058 | 通过 | 0.01080 ± 0.00062 | 0.02411 ± 0.00054 | 0.00133 ± 0.00026 | 0.00517 ± 0.00207 |
| follow_user_num | numeric | False | 0.01476 ± 0.00042 | 通过 | 0.00310 ± 0.00021 | 0.02642 ± 0.00063 | 0.10005 ± 0.00596 | 0.01010 ± 0.00141 |
| short_time_play_user_num | numeric | False | 0.01442 ± 0.00070 | 通过 | 0.02157 ± 0.00027 | 0.00726 ± 0.00113 | 0.00657 ± 0.00227 | 0.01499 ± 0.00152 |
| short_time_play_cnt | numeric | False | 0.01316 ± 0.00113 | 通过 | 0.01956 ± 0.00078 | 0.00676 ± 0.00148 | 0.00452 ± 0.00478 | 0.01256 ± 0.00299 |
| double_click_cnt | numeric | False | 0.01162 ± 0.00072 | 通过 | 0.00094 ± 0.00019 | 0.02231 ± 0.00124 | 0.00035 ± 0.00097 | 0.00621 ± 0.00111 |
| like_cnt | numeric | False | 0.01052 ± 0.00021 | 通过 | 0.00072 ± 0.00010 | 0.02031 ± 0.00032 | 0.00525 ± 0.00184 | 0.00432 ± 0.00149 |
| onehot_feat1 | categorical | False | 0.01014 ± 0.00034 | 通过 | 0.00524 ± 0.00035 | 0.01504 ± 0.00034 | -0.00403 ± 0.00158 | 0.02545 ± 0.00576 |
| like_user_num | numeric | False | 0.00883 ± 0.00030 | 通过 | 0.00196 ± 0.00014 | 0.01571 ± 0.00046 | 0.00148 ± 0.00198 | 0.00320 ± 0.00278 |
| play_progress | numeric | False | 0.00727 ± 0.00041 | 通过 | 0.00711 ± 0.00013 | 0.00743 ± 0.00070 | 0.00335 ± 0.00263 | 0.00183 ± 0.00186 |
| follow_user_num_range | categorical | False | 0.00726 ± 0.00050 | 通过 | 0.00277 ± 0.00000 | 0.01175 ± 0.00099 | 0.01265 ± 0.00395 | 0.00224 ± 0.00189 |
| user_active_degree | categorical | False | 0.00575 ± 0.00016 | 通过 | 0.00439 ± 0.00009 | 0.00710 ± 0.00023 | 0.00726 ± 0.00157 | 0.00267 ± 0.00244 |
| show_user_num | numeric | False | 0.00574 ± 0.00030 | 通过 | 0.00194 ± 0.00014 | 0.00954 ± 0.00047 | 0.00739 ± 0.00321 | 0.00548 ± 0.00433 |
| play_user_num | numeric | False | 0.00574 ± 0.00077 | 通过 | 0.00676 ± 0.00032 | 0.00472 ± 0.00123 | 0.00611 ± 0.00259 | 0.00492 ± 0.00266 |
| long_time_play_cnt | numeric | False | 0.00547 ± 0.00056 | 通过 | 0.00957 ± 0.00034 | 0.00137 ± 0.00077 | 0.00363 ± 0.00270 | -0.00038 ± 0.00070 |
| long_time_play_user_num | numeric | False | 0.00516 ± 0.00044 | 通过 | 0.00767 ± 0.00017 | 0.00265 ± 0.00071 | 0.01058 ± 0.00287 | 0.00860 ± 0.00221 |
| register_days | numeric | False | 0.00444 ± 0.00036 | 通过 | 0.00247 ± 0.00003 | 0.00642 ± 0.00069 | 0.00443 ± 0.00247 | 0.00614 ± 0.00110 |
| video_duration | numeric | False | 0.00437 ± 0.00036 | 通过 | 0.00472 ± 0.00016 | 0.00403 ± 0.00057 | 0.00059 ± 0.00308 | 0.00605 ± 0.00010 |
| friend_user_num | numeric | False | 0.00397 ± 0.00054 | 通过 | 0.00118 ± 0.00007 | 0.00676 ± 0.00101 | 0.00779 ± 0.00627 | 0.00232 ± 0.00114 |
| show_cnt | numeric | False | 0.00336 ± 0.00048 | 通过 | 0.00126 ± 0.00017 | 0.00546 ± 0.00078 | 0.00094 ± 0.00313 | 0.00003 ± 0.00065 |
| reduce_similar_user_num | numeric | False | 0.00310 ± 0.00057 | 通过 | 0.00083 ± 0.00009 | 0.00536 ± 0.00105 | 0.00323 ± 0.00293 | 0.00442 ± 0.00399 |
| onehot_feat7 | categorical | False | 0.00286 ± 0.00046 | 通过 | 0.00273 ± 0.00010 | 0.00300 ± 0.00082 | 0.00010 ± 0.00164 | 0.00193 ± 0.00137 |
| tag | categorical | False | 0.00279 ± 0.00024 | 通过 | 0.00386 ± 0.00012 | 0.00171 ± 0.00035 | 0.00163 ± 0.00118 | 0.00183 ± 0.00177 |
| onehot_feat0 | categorical | False | 0.00264 ± 0.00032 | 通过 | 0.00361 ± 0.00017 | 0.00167 ± 0.00047 | 0.00106 ± 0.00138 | 0.00920 ± 0.00118 |
| play_duration | numeric | False | 0.00254 ± 0.00044 | 通过 | 0.00248 ± 0.00007 | 0.00260 ± 0.00081 | 0.00335 ± 0.00030 | 0.00467 ± 0.00103 |
| cancel_like_cnt | numeric | False | 0.00253 ± 0.00017 | 通过 | 0.00285 ± 0.00017 | 0.00221 ± 0.00017 | 0.00251 ± 0.00209 | 0.00342 ± 0.00138 |
| play_cnt | numeric | False | 0.00248 ± 0.00033 | 通过 | 0.00236 ± 0.00013 | 0.00260 ± 0.00052 | 0.00126 ± 0.00102 | 0.00137 ± 0.00134 |
| friend_user_num_range | categorical | False | 0.00245 ± 0.00027 | 通过 | 0.00108 ± 0.00013 | 0.00381 ± 0.00041 | 0.00599 ± 0.00509 | 0.00047 ± 0.00190 |
| onehot_feat4 | categorical | False | 0.00242 ± 0.00021 | 通过 | 0.00172 ± 0.00012 | 0.00313 ± 0.00031 | -0.00201 ± 0.00255 | 0.00050 ± 0.00210 |
| outsite_share_all_cnt | numeric | False | 0.00239 ± 0.00026 | 通过 | 0.00302 ± 0.00024 | 0.00176 ± 0.00027 | -0.00024 ± 0.00058 | -0.00008 ± 0.00175 |
| cancel_like_user_num | numeric | False | 0.00225 ± 0.00007 | 通过 | 0.00441 ± 0.00009 | 0.00002 ± 0.00010 | -0.00005 ± 0.00194 | 0.00654 ± 0.00053 |
| fans_user_num_range | categorical | False | 0.00214 ± 0.00054 | 通过 | 0.00112 ± 0.00021 | 0.00317 ± 0.00087 | -0.00325 ± 0.00202 | 0.00083 ± 0.00029 |
| follow_cnt | numeric | False | 0.00195 ± 0.00027 | 通过 | 0.00085 ± 0.00010 | 0.00304 ± 0.00044 | 0.04065 ± 0.00364 | 0.00119 ± 0.00303 |
| complete_play_user_num | numeric | False | 0.00192 ± 0.00031 | 通过 | 0.00240 ± 0.00032 | 0.00144 ± 0.00029 | 0.00119 ± 0.00075 | -0.00000 ± 0.00095 |
| reduce_similar_cnt | numeric | False | 0.00179 ± 0.00050 | 通过 | 0.00131 ± 0.00013 | 0.00227 ± 0.00088 | 0.00060 ± 0.00152 | 0.00376 ± 0.00081 |
| click_like_cnt | numeric | False | 0.00171 ± 0.00026 | 通过 | 0.00139 ± 0.00019 | 0.00204 ± 0.00033 | 0.00255 ± 0.00149 | 0.00072 ± 0.00075 |
| onehot_feat10 | categorical | False | 0.00168 ± 0.00042 | 通过 | 0.00161 ± 0.00013 | 0.00175 ± 0.00070 | 0.00024 ± 0.00224 | -0.00061 ± 0.00229 |
| counts | numeric | False | 0.00167 ± 0.00023 | 通过 | 0.00076 ± 0.00002 | 0.00259 ± 0.00044 | -0.00058 ± 0.00139 | 0.00869 ± 0.00338 |
| collect_cnt | numeric | False | 0.00166 ± 0.00024 | 通过 | 0.00037 ± 0.00011 | 0.00295 ± 0.00037 | 0.00094 ± 0.00173 | 0.00233 ± 0.00088 |
| collect_user_num | numeric | False | 0.00155 ± 0.00009 | 通过 | 0.00071 ± 0.00002 | 0.00240 ± 0.00015 | 0.00235 ± 0.00129 | 0.00087 ± 0.00165 |
| cancel_collect_user_num | numeric | False | 0.00145 ± 0.00033 | 通过 | 0.00251 ± 0.00017 | 0.00040 ± 0.00048 | 0.00075 ± 0.00031 | 0.00553 ± 0.00138 |
| direct_comment_cnt | numeric | False | 0.00130 ± 0.00027 | 通过 | 0.00062 ± 0.00002 | 0.00198 ± 0.00053 | 0.00235 ± 0.00233 | 0.00136 ± 0.00076 |
| onehot_feat2 | categorical | False | 0.00125 ± 0.00010 | 通过 | 0.00124 ± 0.00006 | 0.00125 ± 0.00015 | -0.00094 ± 0.00094 | 0.00990 ± 0.00352 |
| reply_comment_cnt | numeric | False | 0.00123 ± 0.00022 | 通过 | 0.00139 ± 0.00004 | 0.00108 ± 0.00040 | 0.00080 ± 0.00091 | 0.00319 ± 0.00118 |
| onehot_feat9 | categorical | False | 0.00123 ± 0.00014 | 通过 | 0.00092 ± 0.00013 | 0.00154 ± 0.00016 | 0.00070 ± 0.00163 | 0.00130 ± 0.00148 |
| follow_user_num1 | numeric | False | 0.00120 ± 0.00011 | 通过 | 0.00117 ± 0.00012 | 0.00122 ± 0.00010 | 0.00421 ± 0.00067 | -0.00191 ± 0.00251 |
| download_cnt | numeric | False | 0.00117 ± 0.00006 | 通过 | 0.00028 ± 0.00008 | 0.00206 ± 0.00004 | 0.00085 ± 0.00082 | 0.00015 ± 0.00113 |
| onehot_feat12 | categorical | False | 0.00110 ± 0.00020 | 待确认 | 0.00166 ± 0.00019 | 0.00054 ± 0.00020 | 0.00141 ± 0.00287 | 0.00481 ± 0.00215 |
| onehot_feat11 | categorical | False | 0.00106 ± 0.00013 | 待确认 | 0.00150 ± 0.00007 | 0.00063 ± 0.00019 | 0.00194 ± 0.00090 | 0.00777 ± 0.00022 |
| comment_like_user_num | numeric | False | 0.00106 ± 0.00009 | 待确认 | 0.00106 ± 0.00007 | 0.00106 ± 0.00011 | -0.00239 ± 0.00081 | 0.00589 ± 0.00119 |
| onehot_feat6 | categorical | False | 0.00103 ± 0.00026 | 待确认 | 0.00125 ± 0.00011 | 0.00080 ± 0.00041 | -0.00015 ± 0.00115 | 0.00363 ± 0.00149 |
| is_video_author | categorical | False | 0.00099 ± 0.00017 | 不通过 | 0.00072 ± 0.00018 | 0.00125 ± 0.00015 | 0.00037 ± 0.00021 | 0.00133 ± 0.00033 |
| direct_comment_user_num | numeric | False | 0.00093 ± 0.00019 | 不通过 | 0.00068 ± 0.00000 | 0.00118 ± 0.00038 | -0.00140 ± 0.00145 | 0.00087 ± 0.00077 |
| hourmin | categorical | False | 0.00090 ± 0.00015 | 不通过 | 0.00139 ± 0.00021 | 0.00041 ± 0.00009 | -0.00027 ± 0.00069 | 0.00027 ± 0.00183 |
| cancel_follow_cnt | numeric | False | 0.00085 ± 0.00025 | 不通过 | 0.00091 ± 0.00025 | 0.00078 ± 0.00026 | 0.00081 ± 0.00130 | 0.00318 ± 0.00178 |
| share_cnt | numeric | False | 0.00083 ± 0.00025 | 不通过 | 0.00050 ± 0.00007 | 0.00117 ± 0.00044 | -0.00265 ± 0.00203 | 0.00386 ± 0.00148 |
| cancel_follow_user_num | numeric | False | 0.00081 ± 0.00019 | 不通过 | 0.00053 ± 0.00007 | 0.00110 ± 0.00032 | 0.00109 ± 0.00067 | 0.00160 ± 0.00063 |
| share_all_cnt | numeric | False | 0.00080 ± 0.00020 | 不通过 | 0.00043 ± 0.00004 | 0.00118 ± 0.00036 | 0.00196 ± 0.00152 | 0.00065 ± 0.00166 |
| reply_comment_user_num | numeric | False | 0.00078 ± 0.00019 | 不通过 | 0.00029 ± 0.00004 | 0.00127 ± 0.00033 | -0.00044 ± 0.00089 | 0.00220 ± 0.00187 |
| complete_play_cnt | numeric | False | 0.00078 ± 0.00016 | 不通过 | 0.00037 ± 0.00002 | 0.00118 ± 0.00030 | 0.00077 ± 0.00070 | 0.00182 ± 0.00138 |
| share_user_num | numeric | False | 0.00078 ± 0.00017 | 不通过 | 0.00053 ± 0.00007 | 0.00102 ± 0.00026 | 0.00013 ± 0.00009 | 0.00095 ± 0.00155 |
| delete_comment_user_num | numeric | False | 0.00074 ± 0.00011 | 不通过 | 0.00077 ± 0.00012 | 0.00070 ± 0.00010 | -0.00007 ± 0.00087 | 0.00108 ± 0.00073 |
| cancel_collect_cnt | numeric | False | 0.00069 ± 0.00014 | 不通过 | 0.00105 ± 0.00006 | 0.00034 ± 0.00021 | 0.00180 ± 0.00158 | 0.00565 ± 0.00185 |
| delete_comment_cnt | numeric | False | 0.00067 ± 0.00009 | 不通过 | 0.00035 ± 0.00004 | 0.00098 ± 0.00014 | 0.00121 ± 0.00112 | -0.00001 ± 0.00035 |
| download_user_num | numeric | False | 0.00066 ± 0.00009 | 不通过 | 0.00020 ± 0.00010 | 0.00111 ± 0.00008 | -0.00066 ± 0.00387 | 0.00383 ± 0.00386 |
| is_live_streamer | categorical | False | 0.00058 ± 0.00015 | 不通过 | 0.00044 ± 0.00006 | 0.00073 ± 0.00023 | 0.00175 ± 0.00128 | 0.00072 ± 0.00081 |
| register_days_range | categorical | False | 0.00058 ± 0.00012 | 不通过 | 0.00035 ± 0.00010 | 0.00080 ± 0.00014 | 0.00216 ± 0.00169 | -0.00140 ± 0.00164 |
| comment_cnt | numeric | False | 0.00056 ± 0.00010 | 不通过 | 0.00057 ± 0.00003 | 0.00056 ± 0.00017 | -0.00136 ± 0.00062 | 0.00298 ± 0.00133 |
| share_all_user_num | numeric | False | 0.00054 ± 0.00028 | 不通过 | 0.00074 ± 0.00020 | 0.00034 ± 0.00036 | -0.00180 ± 0.00118 | 0.00394 ± 0.00287 |
| upload_type | categorical | False | 0.00053 ± 0.00009 | 不通过 | 0.00049 ± 0.00004 | 0.00058 ± 0.00015 | 0.00065 ± 0.00063 | 0.00181 ± 0.00022 |
| date | categorical | False | 0.00047 ± 0.00016 | 不通过 | 0.00037 ± 0.00008 | 0.00057 ± 0.00024 | 0.00034 ± 0.00296 | -0.00235 ± 0.00038 |
| comment_stay_duration | numeric | False | 0.00041 ± 0.00004 | 不通过 | 0.00064 ± 0.00003 | -0.00008 ± 0.00022 | 0.00012 ± 0.00213 | 0.01373 ± 0.00062 |
| comment_user_num | numeric | False | 0.00039 ± 0.00019 | 不通过 | 0.00037 ± 0.00008 | 0.00041 ± 0.00030 | 0.00028 ± 0.00167 | 0.00343 ± 0.00207 |
| onehot_feat16 | categorical | False | 0.00034 ± 0.00007 | 不通过 | 0.00027 ± 0.00009 | 0.00041 ± 0.00005 | -0.00045 ± 0.00060 | 0.00151 ± 0.00128 |
| server_width | numeric | False | 0.00034 ± 0.00008 | 不通过 | 0.00029 ± 0.00012 | 0.00039 ± 0.00004 | -0.00355 ± 0.00063 | 0.00199 ± 0.00057 |
| onehot_feat13 | categorical | False | 0.00033 ± 0.00009 | 不通过 | 0.00016 ± 0.00007 | 0.00049 ± 0.00012 | 0.00195 ± 0.00016 | -0.00018 ± 0.00025 |
| report_user_num | numeric | False | 0.00031 ± 0.00010 | 不通过 | 0.00035 ± 0.00005 | 0.00028 ± 0.00016 | 0.00145 ± 0.00049 | 0.00181 ± 0.00083 |
| server_height | numeric | False | 0.00031 ± 0.00012 | 不通过 | 0.00042 ± 0.00000 | -0.00016 ± 0.00028 | 0.00511 ± 0.00159 | 0.00066 ± 0.00099 |
| onehot_feat14 | categorical | False | 0.00027 ± 0.00011 | 不通过 | 0.00033 ± 0.00002 | -0.00016 ± 0.00026 | 0.00047 ± 0.00100 | 0.00129 ± 0.00090 |
| comment_like_cnt | numeric | False | 0.00025 ± 0.00008 | 不通过 | 0.00042 ± 0.00010 | -0.00002 ± 0.00011 | 0.00129 ± 0.00064 | 0.00440 ± 0.00055 |
| onehot_feat15 | categorical | False | 0.00022 ± 0.00008 | 不通过 | 0.00030 ± 0.00007 | 0.00014 ± 0.00009 | -0.00064 ± 0.00010 | 0.00069 ± 0.00072 |
| fans_user_num | numeric | False | 0.00018 ± 0.00012 | 不通过 | 0.00010 ± 0.00006 | 0.00027 ± 0.00018 | 0.00062 ± 0.00086 | 0.00092 ± 0.00079 |
| upload_dt | categorical | False | 0.00017 ± 0.00010 | 不通过 | 0.00013 ± 0.00006 | 0.00022 ± 0.00014 | -0.00163 ± 0.00012 | -0.00015 ± 0.00064 |
| report_cnt | numeric | False | 0.00013 ± 0.00006 | 不通过 | 0.00018 ± 0.00008 | -0.00003 ± 0.00009 | -0.00010 ± 0.00049 | 0.00103 ± 0.00054 |
| shadow_0 | numeric | True | 0.00008 ± 0.00005 | 不通过 | -0.00006 ± 0.00003 | -0.00001 ± 0.00015 | 0.00079 ± 0.00103 | -0.00081 ± 0.00069 |
| music_type | categorical | False | 0.00007 ± 0.00003 | 不通过 | 0.00010 ± 0.00002 | -0.00001 ± 0.00006 | -0.00003 ± 0.00009 | -0.00030 ± 0.00045 |
| video_type | categorical | False | 0.00003 ± 0.00002 | 不通过 | -0.00000 ± 0.00004 | 0.00002 ± 0.00003 | 0.00052 ± 0.00008 | 0.00040 ± 0.00004 |
| onehot_feat5 | categorical | False | 0.00002 ± 0.00001 | 不通过 | 0.00000 ± 0.00001 | -0.00004 ± 0.00002 | -0.00005 ± 0.00005 | -0.00008 ± 0.00011 |
| onehot_feat17 | categorical | False | 0.00002 ± 0.00002 | 不通过 | -0.00001 ± 0.00001 | -0.00003 ± 0.00003 | -0.00020 ± 0.00022 | -0.00126 ± 0.00095 |
| is_lowactive_period | categorical | False | 0.00000 ± 0.00000 | 不通过 | 0.00000 ± 0.00000 | 0.00000 ± 0.00000 | 0.00000 ± 0.00000 | 0.00000 ± 0.00000 |
| visible_status | categorical | False | 0.00000 ± 0.00000 | 不通过 | 0.00000 ± 0.00000 | 0.00000 ± 0.00000 | 0.00000 ± 0.00000 | 0.00000 ± 0.00000 |

## 噪声参考

{
  "present": true,
  "features": [
    "shadow_0"
  ],
  "overall_mean": 8.207564084080261e-05,
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
