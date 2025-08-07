# Code: Dynamics-Informed Reservoir Computing with Visibility Graphs

Copyright (C) 2024  Charlotte Geier, Dynamics Group (M-14),
Hamburg University of Technology, Hamburg, Germany.
Contact:  [tuhh.de/dyn](https://www.tuhh.de/dyn)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

---
# Overview

Hi there! This repository contains the accompanying code for the 
paper "Dynamics-Informed Reservoir Computing with Visibility Graphs" by Charlotte Geier and Merten Stender. You can use 
this code to reproduce the results presented in the paper, or to perform 
your own studies on functional networks! 
All computation is performed in pure Python.

This code was updated to include revisions. 

**Reference**

Please acknowledge and cite the use of this software and its authors when results are used in publications or published elsewhere. You can use the following reference: 
> C. Geier and M. Stender : **Code for paper Dynamics-Informed Reservoir Computing with Visibility Graphs** (v0.2). Zenodo. [![DOI]()]()


---
# Prerequisites

The code in this repository was tested on 
- WSL: Ubuntu 24.04 on Windows 11
- VS Code 1.102.1
- Python 3.13.1

Requirements are listed in 'requirements.txt' and can be installed via pip 
using `pip install -r requirements.txt`.

The DyRC computations require the computation og averge betweenness, which is not currently implemented in pyReCo. To enable this computation, add the function extract_av_betweenness to utils_networks.

```
# Add more network property extraction functions as needed
def extract_av_betweenness(graph: Union[np.ndarray, nx.Graph, nx.DiGraph]) -> float:
    graph = convert_to_nx_graph(graph)
    betweenness_centrality = nx.betweenness_centrality(graph)
    av_betweenness_centrality = np.mean(list(dict(betweenness_centrality).values()))
    return av_betweenness_centrality
```

---
# Navigation

This code provides the following resources: 
1. The possibility of recreating the figures from data quickly, by using the 
   code and data provided
   1. figure_1.py 
   2. figure_2.py
   3. figure_3.py
   4. figure_4.py
2. The option of reproducing the results from scratch by using the code 
   provided as 
   1. data_generation :: generate Duffing data
   2. DyRC_VG :: reservoir computing ER, DyRC-VG and dense ER
   3. DyRC_VG_16 :: reservoir computing ER, DyRC-VG 16
   4. DyRC_VG_16_compare_sparsity :: additional studies with ER graphs with sparsity comparable to the graphs from VG 16. Attention, this code does not generate VGs, it uses the ones computed in DyRC_VG_16.

### 1. To reproduce figures:
run different figure_xy.py

### 2. To reproduce results from scratch:
1. run data_generation
2. run DyRC_VG and DyRC_VG_16 for every dataset

        

