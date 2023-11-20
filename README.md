# DeepNTax

DeepNTax is a Deep Neural Network (DNN) model that innovatively incorporates taxonomic rank information into its regularization terms. It uses taxonomic rank information for two reasons:
1. The taxonomic rank structure of the data can be directly represented with each layer of the model.
2. Using taxonomic rank rather than a phylogenetic tree makes regularization terms explicit and less affected by the misspecification of the phylogenetic tree.

The regularization matrix in DeepNTax is constructed using two ideas from the taxonomic information:
1. Closer taxonomic information between taxa could indicate that they share similar biological traits.
2. Taxa that differ at a higher taxonomic rank could represent a relatively larger difference between the taxa.

## Prerequisites

To run the DeepNTax model, you need the following files:

- `path_info_DeepNTax.cfg`: A config file containing paths of data files and save directory.
- `network_info_DeepNTax.cfg`: A config file containing information of the model including hyperparameters.
- `otu_table.csv`: An OTU relative abundance table with samples in rows and OTUs in columns.
- `phenotype.csv`: Sample phenotype information.
- `taxonomic_information.csv`: Files containing microbiome phylogenetic information.
- `deepntax`: Source files for the model.

## Running the Code

To run the DeepNTax model, execute the `DeepNTax.py` script with the configuration files as arguments:

```bash
nohup python3 -u DeepNTax.py network_info_DeepNTax.cfg path_info_DeepNTax.cfg > ./DeepNTax_status.txt &
```

This command will start the script in the background and redirect the output to a file named `DeepNTax_status.txt`. The `nohup` command is used to allow the script to continue running even if the terminal is closed.
