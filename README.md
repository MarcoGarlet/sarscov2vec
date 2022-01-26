# sarscov2vec

Realize [Elton et al.](https://arxiv.org/pdf/1903.00415.pdf) pipeline using [Mekni et al.](https://www.mdpi.com/1422-0067/22/14/7714) SARS-CoV-2 viral protease SVM on PubMed Central PMC Open Access articles.


## scheme
<p align="center">
  <img alt="Elton" src="img/EltonPipeline.jpg" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Mekni" src="img/mekni.png" width="35%">
</p>


## project 
<p align="center">
  <img alt="IRArch" src="img/IRArch.png" width="35%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="flowchart" src="img/flowchart.jpg" width="35%">
</p>

ChemDataExtractor is used to identify Chemical Entities validate using PubChemPy and PaDEL-Descriptor software to extract compunds descriptors.  

## description

2-d PCA is used to plot word2vec results following Elton et al. pipeline. 
Moreover, as different approach, elbow method to select optimal out PCA dimension is followed and incremental K-means is applied.

## design

Strategy pattern is followed to dynamically change behavior on different load/store strategies and classifiers. 

<img src="img/StrategyPattern2.png " width="45%"/>

## usage
```console
foo@bar:~/project$ ./build.sh
...
# start padel container
foo@bar:~/project$ ./padel-service/padel-service.sh 
...
# start mongo docker container
foo@bar:~/project$ ./mongo-dock.sh
...
# start project
foo@bar:~/project$ python3 sarscov2vec.py
...

```

Optionally is possible to remove lines in [code/mainProject.py](code/mainProject.py) (commented with "delete this to use FS") to disable usage of MongoDB and use file system to store chemical entities and sentences.
In this case skip start mongo docker container command.


## results

### pca 2-d

PCA 2-d results coloring active compunds against SARS-CoV-2 viral protease.


<table>
  <tr>
    <td><img src="img/result_0_1.png"  alt="40MB" width = "60%"></td>
    <td><img src="img/result_0_3.png" alt="64MB" width = "60%"></td>
  </tr> 
  <tr>
      <td><img src="img/result_0_5.png" alt="190MB" width = "60%"></td>
      <td><img src="img/result_0_7.png"  alt="625MB" width ="60%"></td>
  </tr>
  <tr>
     <td><img src="img/result_0_9.png" alt="747MB" width = "60%"></td>
     <td><img src="img/result_0_11.png" alt="902MB" width = "60%"></td>   
  </tr>
</table>

### optimal PCA out and K-MEANS


<p align="center">
  <img alt="Elton" src="img/PCA_w2vec.png" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Mekni" src="img/K_means.png" width="45%">
</p>

| Cluster Num.      | active CE | CE | words |
| ----------- | ----------- | ----------- | ----------- |
| 0           | 15          | 989         | 89879       |
| 1           | 1           | 92          | 4370        |
| 2           | 0           | 9           | 1272        |


| Coeff.      | value |
| ------------- | ----------- |
| silhouette avg| 0.8479288100264025|
| SSE (k=3)     | 2390           |


| **Term** | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 |
| - | - | - | - | - | - | - | - | - | - | - |
| **covid-19** | ill | psychiatric | hiv-positive | dementia | pandemic | concern | pertain | hemophilia | people | behaviour |

### identified active fragments

<table>

  <tr>
    <td valign="top"><img src="img/carbononitridic-bromide6.png" width="45%"></td>
    <td valign="top"><img src="img/chloroethane5.png" width="45%"></td>
    <td valign="top"><img src="img/chloroform;ethanol7.png" width="45%"></td>
  </tr>
   <tr>
    <td valign="top"><img src="img/chloroform;methanol3.png" width="45%"></td>
    <td valign="top"><img src="img/dichloromethane0.png" width="45%"></td>
    <td valign="top"><img src="img/ethenol2.png" width="45%"></td>
  </tr>
     <tr>
    <td valign="top"><img src="img/furan1.png" width="45%"></td>
    <td valign="top"><img src="img/pentane8.png" width="45%"></td>
    <td valign="top"><img src="img/2-[2-(2-heptoxyethoxy)ethoxy]ethanol4.png" width="45%"></td>
  </tr>
 </table>
