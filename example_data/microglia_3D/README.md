# Channel annotation and intended colocalization analysis

Example filename:

`250703_ID22488_CA1_tdTom_DAPI_IBA1_20x `

## Channel assignment
The channels are assigned as follows:

| Channel | Marker | Interpretation |
|---:|---|---|
| 0 | Cx3cr1-tdTomato | Genetically labelled CX3CR1-positive cells; in brain parenchyma, these are expected to be predominantly microglia. |
| 1 | Iba1 immunostaining | Independent microglia / macrophage-lineage marker. Used here to verify that tdTomato-positive recombined cells are microglia. |
| 2 | DAPI | Nuclear stain. Labels nuclei of all cells. In the CA1 pyramidal cell layer, many DAPI-positive nuclei correspond to neurons, but DAPI is not neuron-specific. |

## Biological interpretation
The experiment most likely uses a tamoxifen-inducible Cre reporter system, for example a Cx3cr1-CreERT2-dependent tdTomato reporter line.

The logic is:

```text
Tamoxifen treatment
     -> activation of CreERT2 in CX3CR1-positive cells
     -> recombination of the reporter locus
     -> tdTomato expression in recombined cells
```

Thus, the tdTomato signal reports cells in which tamoxifen-induced Cre recombination has occurred.

Iba1 does not directly stain tamoxifen-induced Cre recombination. Instead, Iba1 stains microglia / macrophage-lineage cells. In this experiment, Iba1 is used as an independent marker to check whether the tdTomato-positive recombined cells are indeed microglia.

## Colocalization task
The colocalization analysis is between:

```txt
Channel 0: Cx3cr1-tdTomato
Channel 1: Iba1
```

That is:

```text
tdTomato-positive recombined cells  <->  Iba1-positive microglia
```
The central biological question is:

```text
Are the tdTomato-positive, recombined cells also Iba1-positive microglia?
```

## Possible quantitative readouts
Two related but distinct quantities can be computed.

### 1. Specificity of recombination for microglia

```text
N(tdTomato+ ∩ Iba1+) / N(tdTomato+)
```

This asks:

```text
Of all tdTomato-positive recombined cells, what fraction is Iba1-positive?
```

Interpretation:

text A high value means that the recombined tdTomato-positive cells are mostly microglia. 

### 2. Recombination efficiency within the Iba1-positive microglia population

```text
N(tdTomato+ ∩ Iba1+) / N(Iba1+)
```

This asks:

```text
Of all Iba1-positive microglia, what fraction is tdTomato-positive?
```

Interpretation:

```text
A high value means that a large fraction of the Iba1-positive microglia underwent tamoxifen-induced recombination.
```

These two measures are not identical. The first is mainly a measure of specificity, whereas the second is mainly a measure of recombination efficiency.

## Role of DAPI
DAPI is included for anatomical orientation and tissue quality control.

Possible uses of the DAPI channel:

1. Identification of hippocampal anatomy, especially the CA1 pyramidal cell layer.
2. Verification that the tissue section is intact.
3. Detection of folds, tears, damaged regions, or regions with poor imaging quality.
4. Optional support for defining regions of interest.
5. Optional support for cell counting or nuclear segmentation.

DAPI should not be interpreted as a neuron-specific marker. It labels all nuclei. In the CA1 pyramidal layer, many DAPI-positive nuclei are neuronal nuclei, but DAPI alone does not distinguish neurons from glia or other cell types.

