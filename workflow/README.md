# BERT-PSIE workflow

### psie

The folder psie contains the code necessary to fine-tune and deploy the three BERT models that constitute the BERT-PSIE workflow. These models focus on different downstream tasks:

<ol>
<li> <b>Relevancy classification</b>
<li> <b>Named Entity Recognition</b>
<li> <b>Relation classification</b>
</ol>

The three notebooks provided show how to deploy BERT-PSIE for the two extraction tasks explored in the paper, namely the automatic extraction of Curie temperatures and band gaps. The fine tuned models are contained in the models folder.

The notebookes are meant to be run on google colab after creating a copy of this folder on your google drive account. 

Each notebook focusses on each different downstream task. The order in which the notebooks are expected to be executed is:

<ol>
<li> <b>classifier.ipynb</b>
<li> <b>NER.ipynb</b>
<li> <b>relation.ipynb</b>
</ol>

The weights of the fine-tuned models can be found on figshare at the following link:

>
> https://doi.org/10.6084/m9.figshare.23567121.v1
>

The initial corpus given as input to the classifier is expected to be formatted as a JSON file consisting of multiple JSON objects each one including a sentence and its source:

>
> {"sentence": "one sentence extracted from the scientific literature", "source": "10.1000"}
>
>{"sentence": "Another sentence extracted from the scientific literature", "source": "10.1000"}
>
> ...

This format makes it easy to load the file as an HuggingFace dataset which allows working with larger than memory files.

The output of classifier.ipynb will contain only the sentences from the original corpus that have been deemed relevant and it is used as input by NER.ipynb. 

