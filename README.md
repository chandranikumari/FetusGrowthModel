This is supporting data and code for Chandrani Kumari, Gautam I Menon, Leelavati Narlikar, Uma Ram, Rahul Siddharthan, “A predictive model of a growing fetus”. The following files are included

### Data

`seethapathyCleanedData.csv`: Seethapathy cohort data. Patient IDs have been replaced with random IDs, and some maternal data has been removed in the interest of anonymity.

`validationCohort.csv`: Validation cohort data. Patient IDs have been replaced with random IDs, Only ultrasound is included.

### Code and Jupyter notebooks

`FetusGrowth_BWPrediction.py`: function definitions used in the notebooks

`Gompertz_Optimization.ipynb`:  fits Seethapathy cohort to Gompertz equation and gets the optimized value of $t_0$ and $c$ globally over all fetuses, and $A$ individually for each fetus.

`Gompertz_Optimization+Plots.ipynb`: figure 2 and figure 3 in manuscript + hypergeometric test.

`Prediction_LR.ipynb`: birth weight prediction using linear regression

`BWPridiction_plot.ipynb`: plots figures related to birth weight prediction

