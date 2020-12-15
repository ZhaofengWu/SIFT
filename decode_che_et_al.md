1. The original parser model and code are at https://github.com/DreamerDeo/HIT-SCIR-CoNLL2019. However, when adapting to a more general corpus than CoNLL 2019 (i.e. GLUE), there were some issues with the original code. I made some fixes at my fork https://github.com/ZhaofengWu/HIT-SCIR-CoNLL2019. Some of these fixes have been incorporated into the original repo after I opened some issues there, but not all. Nevertheless, they may have also fixed some other things. So it's up to you which one you use.

2. Get mtool at https://github.com/cfmrp/mtool, and checkout `f5941ad` to be consistent with its original CoNLL-19 release.

3. Convert the dataset into a one sentence per line format with the following command. `${TASK_NAME}` can be `cola`, `mnli`, etc. The output files are `${SPLIT}.lines`.

    ```bash
    python scripts/dataset_to_lines.py ${TASK_NAME} ${DATASET_DIR} ${DATASET_DIR}
    ```

Example output format (the whitespaces are a tab):

    ```
    0       The sailors rode the breeze clear of the rocks.
    1       The weights made the rope stretch over the pulley.
    ```

We create the sentence IDs ourselves.

4. Use the official CoNLL 2019 pipeline, consisting of REPP and UDPipe, to annotate `${SPLIT}.lines` into `${SPLIT}.conllu`.

Example output format (the whitespaces are a tab):

    ```
    #0
    1       The     the     DET     DT      _       2       det     _       TokenRange=0:3
    2       sailors sailor  NOUN    NNS     _       3       nsubj   _       TokenRange=4:11
    3       rode    ride    VERB    VBD     _       0       root    _       TokenRange=12:16
    4       the     the     DET     DT      _       5       det     _       TokenRange=17:20
    5       breeze  breeze  NOUN    NN      _       3       obj     _       TokenRange=21:27
    6       clear   clear   ADJ     JJ      _       3       xcomp   _       TokenRange=28:33
    7       of      of      ADP     IN      _       9       case    _       TokenRange=34:36
    8       the     the     DET     DT      _       9       det     _       TokenRange=37:40
    9       rocks   rock    NOUN    NNS     _       6       obl     _       TokenRange=41:46
    10      .       .       PUNCT   .       _       3       punct   _       TokenRange=46:47

    #1
    1       The     the     DET     DT      _       2       det     _       TokenRange=0:3
    2       weights weight  NOUN    NNS     _       3       nsubj   _       TokenRange=4:11
    3       made    make    VERB    VBD     _       0       root    _       TokenRange=12:16
    4       the     the     DET     DT      _       5       det     _       TokenRange=17:20
    5       rope    rope    NOUN    NN      _       6       nsubj   _       TokenRange=21:25
    6       stretch stretch VERB    VB      _       3       ccomp   _       TokenRange=26:33
    7       over    over    ADP     IN      _       9       case    _       TokenRange=34:38
    8       the     the     DET     DT      _       9       det     _       TokenRange=39:42
    9       pulley  pulley  NOUN    NN      _       6       obl     _       TokenRange=43:49
    10      .       .       PUNCT   .       _       3       punct   _       TokenRange=49:50
    ```

As far as we know, this pipeline has not yet been made publicly accessible, but you can use other tools for this conversion. The only tasks involved here are reversible tokenization (i.e., with character offset; important!) and dependency parsing. If you do this with other tools, the input to the parser will perhaps have some slight distributional differences from its original training data that was annotated with the CoNLL 2019 pipeline. Nevertheless, since these two tasks are relatively simple, the differences shouldn't be dramatic. We also provide pre-parsed GLUE graphs using the CoNLL 2019 pipeline; see the main [README.md](README.md).

5. Generate the semantic graphs `${SPLIT}.${FORMALISM}.rdf` and the associated metadata `${SPLIT}.${FORMALISM}.metadata` (i.e., anchors and other features) with the following command. It currently assumes that the HIT-SCIR-CoNLL2019 directory and the mtool directory are siblings to this project directory.

    ```bash
    bash scripts/decode_che_et_al.sh ${DATASET_DIR}
    ```

6. The main script will look for the `${SPLIT}.${FORMALISM}.rdf` and `${SPLIT}.${FORMALISM}.metadata` files within the dataset directory, e.g., `data/glue_data/CoLA/train.dm.rdf`, etc.
