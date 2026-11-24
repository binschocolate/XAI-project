# ROAR 결과 요약

## 1. ROAR 점수 (지운 비율 10–70% 평균 test 정확도, 낮을수록 좋음; seed 평균)

```
method                      gradcam     ig   rise  random
arch     model    baseline                               
resnet18 clean    mean        0.608  0.703  0.731   0.743
                  trainset    0.551  0.601  0.612   0.670
         shortcut mean        0.557  0.586  0.663   0.613
                  trainset    0.528  0.551  0.585   0.592
resnet50 clean    mean        0.629  0.744  0.727   0.767
                  trainset    0.564  0.627  0.630   0.692
         shortcut mean        0.582  0.601  0.661   0.641
                  trainset    0.531  0.557  0.580   0.607
```

지우기 0% 재학습 정확도 (seed 평균):

```
arch      model   
resnet18  clean       0.892
          shortcut    0.752
resnet50  clean       0.920
          shortcut    0.796
```

## 2. Random 보다 나은 비율 (run 수 기준)

```
method             gradcam   ig  rise
arch     baseline                    
resnet18 mean          1.0  1.0  0.33
         trainset      1.0  1.0  0.67
resnet50 mean          1.0  1.0  0.33
         trainset      1.0  1.0  0.83
```

## 3. baseline(mean vs trainset) 에 따른 순위 변화

```
                  run       rank_mean   rank_trainset  flip_mean_vs_trainset  tau_mean_vs_trainset
resnet18_rho050_seed0 gradcam>ig>rise gradcam>ig>rise                  False              1.000000
resnet18_rho050_seed1 gradcam>ig>rise gradcam>ig>rise                  False              1.000000
resnet18_rho050_seed2 gradcam>rise>ig gradcam>ig>rise                   True              0.333333
resnet18_rho095_seed0 gradcam>ig>rise gradcam>ig>rise                  False              1.000000
resnet18_rho095_seed1 gradcam>ig>rise gradcam>ig>rise                  False              1.000000
resnet18_rho095_seed2 gradcam>ig>rise gradcam>ig>rise                  False              1.000000
resnet50_rho050_seed0 gradcam>rise>ig gradcam>rise>ig                  False              1.000000
resnet50_rho050_seed1 gradcam>rise>ig gradcam>rise>ig                  False              1.000000
resnet50_rho050_seed2 gradcam>ig>rise gradcam>ig>rise                  False              1.000000
resnet50_rho095_seed0 gradcam>ig>rise gradcam>ig>rise                  False              1.000000
resnet50_rho095_seed1 gradcam>ig>rise gradcam>ig>rise                  False              1.000000
resnet50_rho095_seed2 gradcam>ig>rise gradcam>ig>rise                  False              1.000000
```

S_roar = 1 − τ(mean, trainset), D_roar = shortcut − clean (같은 arch·seed):

```
model          clean  shortcut  D_roar
arch     seed                         
resnet18 0     0.000       0.0   0.000
         1     0.000       0.0   0.000
         2     0.667       0.0  -0.667
resnet50 0     0.000       0.0   0.000
         1     0.000       0.0   0.000
         2     0.000       0.0   0.000
```

## 4. Deletion(MIF) 순위와의 일치 (Kendall τ-b, 1 = 같은 순위)

```
                   tau_roar_vs_mif_mean  tau_roar_vs_mif_trainset
arch     model                                                   
resnet18 clean                   -0.111                     0.111
         shortcut                -0.333                    -0.333
resnet50 clean                    0.111                     0.556
         shortcut                -0.333                    -0.111
```
