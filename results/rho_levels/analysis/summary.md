# ρ 수준 경향 분석 (dose-response)

## SRI (ρ 별, seed 평균·최소·최대)

```
           mean                         min                         max                     
rho        0.50   0.75   0.95   0.99   0.50   0.75   0.95   0.99   0.50   0.75   0.95   0.99
arch                                                                                        
resnet18  0.014  0.121  0.409  0.607  0.012  0.120  0.400  0.554  0.015  0.122  0.421  0.637
resnet50  0.005  0.074  0.344  0.573  0.004  0.067  0.337  0.558  0.007  0.082  0.350  0.585
```

## 주 분석: 기법 3개, MIF, 전체 800장 — S 와 SRI 의 관계

```
  kind     arch methods metric scope  spearman_mean spearman_pos  slope_per_SRI  slope_lo  slope_hi  S_rho0.5  S_rho0.75  S_rho0.95  S_rho0.99
rank_S resnet18       3    mif   all          0.555          3/3          0.518     0.280     0.817     0.178      0.089      0.222      0.489
rank_S resnet50       3    mif   all         -0.298          1/3         -0.178    -0.505     0.250     0.178      0.178      0.089      0.089
```

## 순위 민감도 S 경향 (전체)

```
  kind     arch methods metric       scope  spearman_mean spearman_pos  slope_per_SRI  slope_lo  slope_hi  S_rho0.5  S_rho0.75  S_rho0.95  S_rho0.99
rank_S resnet18       3    lif     aligned         -0.576          0/3         -0.318    -0.585    -0.157     0.133      0.222      0.000      0.000
rank_S resnet18       3    lif         all         -0.718          0/3         -0.418    -0.589    -0.188     0.222      0.267      0.089      0.000
rank_S resnet18       3    lif conflicting         -0.504          0/3         -0.140    -0.451     0.087     0.267      0.311      0.311      0.178
rank_S resnet18       3    mif     aligned          0.898          3/3          1.194     0.504     1.321     0.178      0.356      0.844      0.844
rank_S resnet18       3    mif         all          0.555          3/3          0.518     0.280     0.817     0.178      0.089      0.222      0.489
rank_S resnet18       3    mif conflicting          0.223          2/3          0.084    -0.280     0.302     0.089      0.222      0.133      0.178
rank_S resnet18       3    srg     aligned         -0.775          0/3         -0.068    -0.135     0.099     0.356      0.356      0.356      0.311
rank_S resnet18       3    srg         all         -0.775          0/3         -0.137    -0.194     0.104     0.267      0.267      0.267      0.178
rank_S resnet18       3    srg conflicting         -0.835          0/3         -0.299    -0.471    -0.227     0.178      0.089      0.000      0.000
rank_S resnet18       5    lif     aligned         -0.800          0/3         -0.444    -0.607    -0.299     0.400      0.387      0.213      0.160
rank_S resnet18       5    lif         all         -0.800          0/3         -0.531    -0.662    -0.403     0.387      0.320      0.133      0.080
rank_S resnet18       5    lif conflicting         -0.733          0/3         -0.330    -0.456    -0.111     0.360      0.227      0.093      0.160
rank_S resnet18       5    mif     aligned          0.896          3/3          0.705     0.365     0.810     0.200      0.320      0.587      0.613
rank_S resnet18       5    mif         all          0.172          2/3          0.085    -0.032     0.218     0.187      0.173      0.187      0.240
rank_S resnet18       5    mif conflicting         -0.333          1/3         -0.097    -0.266     0.007     0.147      0.173      0.173      0.080
rank_S resnet18       5    srg     aligned          0.267          2/3          0.064    -0.027     0.221     0.253      0.307      0.280      0.320
rank_S resnet18       5    srg         all          0.333          2/3          0.056    -0.007     0.207     0.253      0.213      0.213      0.293
rank_S resnet18       5    srg conflicting         -0.304          1/3         -0.167    -0.338    -0.004     0.173      0.120      0.053      0.080
rank_S resnet50       3    lif     aligned         -0.164          1/3         -0.104    -0.291     0.136     0.133      0.089      0.000      0.089
rank_S resnet50       3    lif         all          0.387          2/3          0.131     0.006     0.356     0.089      0.311      0.178      0.267
rank_S resnet50       3    lif conflicting          0.211          3/3          0.103    -0.188     0.440     0.000      0.311      0.178      0.178
rank_S resnet50       3    mif     aligned          1.000          3/3          1.364     0.981     1.700     0.222      0.667      0.933      1.111
rank_S resnet50       3    mif         all         -0.298          1/3         -0.178    -0.505     0.250     0.178      0.178      0.089      0.089
rank_S resnet50       3    mif conflicting         -0.775          0/3         -0.211    -0.715    -0.104     0.178      0.000      0.000      0.000
rank_S resnet50       3    srg     aligned          0.407          2/3          0.133    -0.312     0.357     0.311      0.267      0.356      0.356
rank_S resnet50       3    srg         all         -0.258          1/3         -0.014    -0.365     0.157     0.356      0.267      0.311      0.311
rank_S resnet50       3    srg conflicting         -0.931          0/3         -0.653    -0.653    -0.443     0.356      0.267      0.000      0.000
rank_S resnet50       5    lif     aligned         -0.509          1/3         -0.256    -0.402     0.065     0.333      0.333      0.200      0.213
rank_S resnet50       5    lif         all         -0.376          1/3         -0.288    -0.336    -0.073     0.347      0.373      0.147      0.240
rank_S resnet50       5    lif conflicting         -0.783          0/3         -0.297    -0.423     0.065     0.307      0.320      0.107      0.187
rank_S resnet50       5    mif     aligned          0.744          3/3          0.543     0.294     0.767     0.200      0.480      0.613      0.573
rank_S resnet50       5    mif         all         -0.304          1/3         -0.161    -0.307     0.108     0.213      0.187      0.133      0.120
rank_S resnet50       5    mif conflicting         -0.467          0/3         -0.153    -0.470    -0.065     0.133      0.133      0.160      0.027
rank_S resnet50       5    srg     aligned          0.105          1/3         -0.015    -0.309     0.125     0.307      0.360      0.333      0.320
rank_S resnet50       5    srg         all          0.075          1/3          0.021    -0.243     0.111     0.347      0.333      0.280      0.373
rank_S resnet50       5    srg conflicting         -0.611          0/3         -0.321    -0.526    -0.182     0.387      0.213      0.107      0.173
```

## 점수 변동 SD 경향

```
    kind     arch    methods metric scope  spearman_mean spearman_pos  slope_per_SRI  slope_lo  slope_hi  S_rho0.5  S_rho0.75  S_rho0.95  S_rho0.99
score_sd resnet18    gradcam    lif   all         0.9333          3/3         0.0482       NaN       NaN    0.0131     0.0200     0.0381     0.0402
score_sd resnet18    gradcam    mif   all         0.8000          3/3         0.0303       NaN       NaN    0.0137     0.0115     0.0232     0.0299
score_sd resnet18    gradcam    srg   all         0.8000          3/3         0.0504       NaN       NaN    0.0198     0.0233     0.0504     0.0452
score_sd resnet18         ig    lif   all        -0.8667          0/3        -0.0530       NaN       NaN    0.0516     0.0370     0.0184     0.0197
score_sd resnet18         ig    mif   all         0.4667          2/3         0.0186       NaN       NaN    0.0226     0.0113     0.0236     0.0293
score_sd resnet18         ig    srg   all        -0.9333          0/3        -0.0346       NaN       NaN    0.0353     0.0344     0.0204     0.0165
score_sd resnet18       lime    lif   all         0.9333          3/3         0.0308       NaN       NaN    0.0254     0.0300     0.0422     0.0426
score_sd resnet18       lime    mif   all         0.8000          3/3         0.0315       NaN       NaN    0.0274     0.0282     0.0412     0.0450
score_sd resnet18       lime    srg   all         0.8000          3/3         0.0422       NaN       NaN    0.0445     0.0535     0.0729     0.0684
score_sd resnet18       rise    lif   all         0.8667          3/3         0.0234       NaN       NaN    0.0338     0.0395     0.0480     0.0476
score_sd resnet18       rise    mif   all         0.8667          3/3         0.0313       NaN       NaN    0.0327     0.0294     0.0430     0.0494
score_sd resnet18       rise    srg   all         0.6000          3/3         0.0295       NaN       NaN    0.0569     0.0621     0.0787     0.0729
score_sd resnet18 smoothgrad    lif   all         0.9333          3/3         0.0256       NaN       NaN    0.0131     0.0174     0.0246     0.0283
score_sd resnet18 smoothgrad    mif   all         0.8000          3/3         0.0341       NaN       NaN    0.0133     0.0129     0.0221     0.0331
score_sd resnet18 smoothgrad    srg   all         0.4667          3/3         0.0092       NaN       NaN    0.0101     0.0084     0.0057     0.0174
score_sd resnet50    gradcam    lif   all         1.0000          3/3         0.0699       NaN       NaN    0.0098     0.0158     0.0397     0.0482
score_sd resnet50    gradcam    mif   all         0.9333          3/3         0.0508       NaN       NaN    0.0064     0.0079     0.0175     0.0359
score_sd resnet50    gradcam    srg   all         0.8667          3/3         0.0716       NaN       NaN    0.0110     0.0176     0.0465     0.0491
score_sd resnet50         ig    lif   all         0.1333          2/3         0.0104       NaN       NaN    0.0633     0.0609     0.0497     0.0725
score_sd resnet50         ig    mif   all         0.2000          3/3         0.0376       NaN       NaN    0.0310     0.0276     0.0260     0.0542
score_sd resnet50         ig    srg   all        -0.9333          0/3        -0.0228       NaN       NaN    0.0325     0.0338     0.0254     0.0210
score_sd resnet50       lime    lif   all         1.0000          3/3         0.0499       NaN       NaN    0.0203     0.0242     0.0386     0.0484
score_sd resnet50       lime    mif   all         0.9333          3/3         0.0420       NaN       NaN    0.0118     0.0123     0.0205     0.0360
score_sd resnet50       lime    srg   all         0.9333          3/3         0.0537       NaN       NaN    0.0305     0.0352     0.0526     0.0605
score_sd resnet50       rise    lif   all         1.0000          3/3         0.0492       NaN       NaN    0.0255     0.0302     0.0480     0.0523
score_sd resnet50       rise    mif   all         0.8000          3/3         0.0329       NaN       NaN    0.0168     0.0162     0.0232     0.0354
score_sd resnet50       rise    srg   all         0.8667          3/3         0.0460       NaN       NaN    0.0388     0.0443     0.0653     0.0630
score_sd resnet50 smoothgrad    lif   all         1.0000          3/3         0.0374       NaN       NaN    0.0113     0.0159     0.0263     0.0329
score_sd resnet50 smoothgrad    mif   all         0.9333          3/3         0.0595       NaN       NaN    0.0107     0.0108     0.0236     0.0444
score_sd resnet50 smoothgrad    srg   all         0.8667          3/3         0.0185       NaN       NaN    0.0032     0.0078     0.0074     0.0161
```
