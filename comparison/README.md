# BERT-PSIE vs ChemDataExtractor Direct Comparison

A total of 200 abstracts were manually annotated for the case of compound-Curie temperature extraction and separately for compound-band gap extraction. To obtain this corpus of test samples, unique abstracts were taken from the arXiv. Care was taken to guarantee that there was no overlap between the abstract used for the fine-tuning or validation of the BERT models and the ones used in these test sets. The new test abstracts were obtained by running a keyword search on a sample of unused
abstracts in the database. 
Both BERT-PSIE and ChemDataExtractor were run on this test set. A record in the extracted database was deemed true positive only if all entities in the target compound-property pair were present and matched the manual annotation. The number of true positives, false positives and false negatives for the extraction tasks were manually counted for each property extraction, allowing for the calculation of the precision, recall and F1 score for each model. Given the nature of this comparison we decided to use arXiv abstracts so that we could release them. This folder contains the abstracts used, togheter with our manual labels and the extractions performed by ChemDataExtractor and our BERT-PSIE workflow.


### Abstracts

The abstracts_prop.json files contain the aeXiv abstract used togeter with their id

### Extractions

All the csv file contain extracted compound-property pairs togheter with the abstract number from which they come from with respect to the abstracts json files

<ol>
<li> <b>manual:</b> compound-property pairs extracted manually
<li> <b>chem_data:</b> compound-property pairs extracted with ChemDataExtractor
<li> <b>Bert:</b> compound-property pairs extracted with BERT-PSIE
</ol>

