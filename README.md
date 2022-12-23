# FetusGrowthModel

This is the supporting information for the manuscript titled **A predictive model of a growing fetus**. The descriptions of each file used for plotting, optimization, and predictions are provided below.

1. **seethapathyCleanedData.csv** ---- "Seethapathy cohort"

2. **validationCohort.csv** ------Validation set

3. python file named **FetusGrowth_BWPrediction.py** contains all the functions used for the analysis.

4. **Gompertz_Optimization.ipynb** fits "Seethapathy cohort" to Gompertz equation and get the optimized value of $t_0$ and $c$ globally over all fetuses, and $A$ individually for each fetus.

5.  **Gompertz_Optimization+Plots.ipynb** does hypergeomtric test and plots figure 2 and figure 3 in manuscript.

6. **IntergrowthData_optimization.ipynb** and **IntergrowthData+plotting.ipynb** Compared Gompertz equation with INTERGROWTH fetus growth model.

7. **BWPridiction_LR.ipynb** birth weight prediction using linear regression

8. **BWPridiction_plot.ipynb** plots all the figures related to birth weight prediction

