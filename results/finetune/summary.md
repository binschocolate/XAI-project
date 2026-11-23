# 마스킹 적응 추가 학습 실험 요약 (ResNet-18, seed 0–2 평균)

## 1. Shortcut 의존성 유지 확인 (test 5,794장)

```
                   avg_acc        worst_group_acc        group_gap           sri       
                      mean    std            mean    std      mean    std   mean    std
model    cond                                                                          
clean    control     0.937  0.002           0.803  0.017     0.010  0.007  0.012  0.004
         maskadapt   0.931  0.003           0.754  0.032     0.005  0.007  0.004  0.002
         original    0.936  0.001           0.787  0.022     0.015  0.001  0.014  0.002
shortcut control     0.803  0.004           0.368  0.020     0.418  0.004  0.412  0.004
         maskadapt   0.834  0.008           0.261  0.048     0.384  0.014  0.377  0.009
         original    0.812  0.006           0.332  0.004     0.413  0.012  0.409  0.011
```

## 2. 순위 민감도 S — 새 attribution 으로 평가 (최종 결과)

```
cond                         original  control  maskadapt
metric scope       model                                 
mif    aligned     clean        0.178    0.178      0.178
                   shortcut     0.844    0.889      0.711
       all         clean        0.178    0.178      0.089
                   shortcut     0.222    0.400      0.089
       conflicting clean        0.089    0.000      0.000
                   shortcut     0.133    0.089      0.133
srg    aligned     clean        0.356    0.311      0.311
                   shortcut     0.356    0.356      0.356
       all         clean        0.267    0.267      0.267
                   shortcut     0.267    0.267      0.267
       conflicting clean        0.178    0.178      0.267
                   shortcut     0.000    0.000      0.000
```

### D = S_shortcut − S_clean — 새 attribution 으로 평가 (최종 결과)

```
                      mean                    pos_seeds                   
cond               control maskadapt original   control maskadapt original
metric scope                                                              
mif    aligned       0.711     0.533    0.667         3         3        3
       all           0.222     0.000    0.044         2         1        1
       conflicting   0.089     0.133    0.044         1         1        1
srg    aligned       0.044     0.044    0.000         1         1        0
       all           0.000     0.000    0.000         0         0        0
       conflicting  -0.178    -0.267   -0.178         0         0        0
```

## 2. 순위 민감도 S — 원래 attribution·target 고정 (보조 진단)

```
cond                         original  control  maskadapt
metric scope       model                                 
mif    aligned     clean        0.178    0.178      0.311
                   shortcut     0.844    0.889      0.978
       all         clean        0.178    0.178      0.178
                   shortcut     0.222    0.267      0.267
       conflicting clean        0.089    0.089      0.089
                   shortcut     0.133    0.133      0.000
srg    aligned     clean        0.356    0.311      0.222
                   shortcut     0.356    0.356      0.311
       all         clean        0.267    0.267      0.267
                   shortcut     0.267    0.267      0.267
       conflicting clean        0.178    0.178      0.267
                   shortcut     0.000    0.000      0.000
```

### D = S_shortcut − S_clean — 원래 attribution·target 고정 (보조 진단)

```
                      mean                    pos_seeds                   
cond               control maskadapt original   control maskadapt original
metric scope                                                              
mif    aligned       0.711     0.667    0.667         3         3        3
       all           0.089     0.089    0.044         2         1        1
       conflicting   0.044    -0.089    0.044         1         0        1
srg    aligned       0.044     0.089    0.000         1         1        0
       all           0.000     0.000    0.000         0         0        0
       conflicting  -0.178    -0.267   -0.178         0         0        0
```
