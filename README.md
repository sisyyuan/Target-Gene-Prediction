# Target Gene Predictions of CRMs Revels Their Functional Types 
## The training datasets for predicting funcitonal states of CRMs (file size too big, split into two smaller files)
   - TrainData_1.zip
   - TrainData_2.zip 

## The code and models for training, evaluation, and testing:
   - Train_evaluate_model.py
   - LogisticRegression-CA.joblib

## The code for predicting active CRM in the whole genome:
   - Predict-ENCODE.py

## Predicted active CRM in the heart-right-ventricle cells
   - human-donorsENCDO967KID-heart-right-ventricle.CRM.minmax.zip
   - human-donorsENCDO967KID-heart-right-ventricle.CRM.minmaxLogisticRegression.predict.zip

## Mann whitney U test:
   - MWU-test.py

## Predictions of Regulations
   columns: CRM(chr-start-end) Gene(chr-start-end) q-value
   - up regulaitons: Enhancer_Hic_Validated_Regulations_2000.uniq.HiC20.bed.zip
   - down regulations: Silencer_Hic_Validated_Regulations_2000.uniq.HiC20.bed.zip
