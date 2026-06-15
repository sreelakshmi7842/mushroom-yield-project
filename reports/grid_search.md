\\# Random Forest Hyperparameter Tuning







\\## Parameter Grid







\\### n\\\_estimators







Values tested:







\\- 50



\\- 100



\\- 200







Purpose:







Controls the number of trees in the forest.







\\### max\\\_depth







Values tested:







\\- None



\\- 8



\\- 16







Purpose:







Limits tree complexity and reduces overfitting.







\\### min\\\_samples\\\_leaf







Values tested:







\\- 1



\\- 3



\\- 5







Purpose:







Controls minimum samples required in a leaf node.







\\---







\\## Cross Validation







TimeSeriesSplit(n\\\_splits=3)







Only training data was used.







No test data participated in tuning.







\\---







\\## Best Parameters







(Insert output here)







\\---







\\## Test Performance







| Metric | Value |



|----------|----------|



| MAE | |



| RMSE | |



| R² | |







\\---







\\## Runtime







(Insert runtime)







\\---







\\## Conclusion







The tuned Random Forest model was selected using TimeSeriesSplit cross-validation and evaluated once on the hold-out test set.



