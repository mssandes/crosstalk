This repository was created to share the code and dataset used to produce the results in the paper "Crosstalk Mitigation in an ATLAS-like  High Energy Liquid Argon Calorimeter using Artificial Neural Networks", submitted to NIMA.

# Crosstalk (XT) in a Liquid Argon Calorimeter (LAr)

Crosstalk is a well-known issue in electronic circuits. In Liquid Argon (LAr) calorimeters, the high granularity of sensor cells and the high particle energies can produce parasitic effects that distort the signal of interest. One such effect is crosstalk (XT), a spurious signal induced by neighboring cells.

<p align="center">
  <b>Fig. 1.</b> Cell signals - At left, cell signal, XT L, and C, and noise contribution. At right the details for distortion caused by XT.
</p>
<table align="center">
  <tr>
    <td align="center">
      <img src="XTdefinitions.png" width="400"><br>
    </td>
  </tr>
</table>


This module is based on particle physics models and is designed to compute inductive and capacitive crosstalk contributions within a cluster cell of a Liquid Argon Calorimeter. It is part of the [Lorenzetti](https://github.com/lorenzetti-ufrj-br/lorenzetti) framework for high-energy physics studies.

(a) Standard cluster used for energy and time estimation. (b) Cell addressing scheme used in this module, where the red cell is the hottest cell and the yellow cells represent first neighbors, corresponding to first-order crosstalk. (c) Inductive (L) and capacitive (C) coupling in the cell of interest.

<p align="center">
  <b>Fig. 2.</b> (a) Standard Cluster, (b) cluster addressing scheme, and (c) LC coupling.
</p>
<table align="center">
  <tr>
    <td align="center">
      <img src="Cluster.png" width="250"><br>
      <b>(a)</b> 
    </td>
    <td align="center">
      <img src="clusterAdressing.png" width="150"><br>
      <b>(b)</b> 
    </td>
    <td align="center">
      <img src="LCcoupling_sf.png" width="220"><br>
      <b>(c)</b> 
    </td>
  </tr>
</table>

A 3×3 moving window (Fig. 3) is applied over the 7×7 cluster to compute first-neighbor XT contributions for each cell of interest, producing a composite signal that includes both noise and crosstalk effects.

<p align="center">
  <b>Fig. 3.</b> Relative capacitive and inductive XT clusters production with respect a interest cell.
</p>
<table align="center">
  <tr>
    <td align="center">
      <img src="RelatCluster.png" width="800"><br>
    </td>
  </tr>
</table>

An example of data generated with this module is available [here](https://github.com/mssandes/crosstalk/blob/main/generateSignals.ipynb)

Main function to XT studies:
-     cellFunction(t)
- Generates the g(t) signal for an LAr calorimeter cell at sampling time t.
-      XTalk(t)
- Computes capacitive or inductive crosstalk (XT) signals for a neighboring cell at sampling time t.
-     relativeClusters(Etruth)
- Produces relative capacitive and inductive clusters based on the true deposited energy.
