This is supporting data and code for Chandrani Kumari, Gautam I Menon, Leelavati Narlikar, Uma Ram, Rahul Siddharthan, “A predictive model of a growing fetus”. The following files are included


`seethapathyCleanedData.csv`: Seethapathy cohort data. Patient IDs have been replaced with random IDs, and some maternal data has been removed in the interest of anonymity.

`validationCohort.csv`: Validation cohort data. Patient IDs have been replaced with random IDs, Only ultrasound is included.

3. python file named **FetusGrowth_BWPrediction.py** contains all the functions used for the analysis.

4. **Gompertz_Optimization.ipynb** fits "Seethapathy cohort" to Gompertz equation and get the optimized value of $t_0$ and $c$ globally over all fetuses, and $A$ individually for each fetus.

5.  **Gompertz_Optimization+Plots.ipynb** does hypergeomtric test and plots figure 2 and figure 3 in manuscript.

6. **IntergrowthData_optimization.ipynb** and **IntergrowthData+plotting.ipynb** Compared Gompertz equation with INTERGROWTH fetus growth model.

7. **BWPridiction_LR.ipynb** birth weight prediction using linear regression

8. **BWPridiction_plot.ipynb** plots all the figures related to birth weight prediction

