# BERT-PSIE-TC
A dataset of Curie temperatures automatically extracted from scientific literature with the use of BERT-PSIE.

<img src="./images/hist_comparison.png" width=700 >

The script train_RF.py trains a random forest predictor for the Curie Temperature over the Tc dataset. The model can be used to screen ferroelettric candidates based on their predicted Curie temperature.

| <img src="./images/screening.png" width=350 > |
|:--:|
|	Violin plots showing the TC distributions of the compounds
screened using a RF model trained on the BERT-PSIE data and com-
pared with the manually extracted values (top panel). The dashed
line is the parity line highlighting how the median of the screened
distribution increases as the screening threshold increases. Despite a
low recall, the precision is high enough to select compounds likely
to have a TC higher than a given threshold. The screening is done
on compounds not present in the training set of the RF. |


### References

