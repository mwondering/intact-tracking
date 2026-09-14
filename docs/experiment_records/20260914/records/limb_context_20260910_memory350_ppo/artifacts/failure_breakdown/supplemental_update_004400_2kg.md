# Memory350 失败触发项：update_004400_2kg

统计已记录轨迹中的终止触发项。一次失败可能同时触发多项，JSON 保留组合计数；这些计数不能说明关闭某项后策略会恢复还是继续失稳。

| 场景 | 策略 | Motions | 失败次数 | Anchor z 位置 | Anchor 姿态 | EE z 位置 |
|---|---|---:|---:|---:|---:|---:|
| all_2_cold | baseline | 512 | 2 | 0 | 0 | 2 |
| all_2_cold | film | 512 | 5 | 0 | 0 | 5 |
| all_2_warm | baseline | 512 | 2 | 0 | 0 | 2 |
| all_2_warm | film | 512 | 7 | 1 | 0 | 7 |
