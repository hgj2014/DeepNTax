# DeepNTax

This project contains the code for the DeepNTax model. 

DeepNTax is a DNN model that innovatively incorporates taxonomic rank information into its regularization terms. Taxonomic rank information was also used in defining the structure of the model. Taxonomic rank is used in DeepNTax for two reasons. First, the taxonomic rank structure of the data can be directly represented with each layer of the model. Secondly, using taxonomic rank rather than a phylogenetic tree makes regularization terms explicit and less affected by the misspecifi-cation of the phylogenetic tree. The taxonomic information was encompassed in regularization for the proposed model. 

In DeepNTax, the regularization matrix is constructed using the two ideas from the taxonomic information. First, closer taxonomic information between taxa could indicate that they share similar biologi-cal traits. Second, taxa that differ at a higher taxonomic rank could represent a relatively larger difference between the taxa. 


## Running the Code
To run the code, you need following files. 

* path_info_DeepNTax.cfg: config file containing paths of data files and save directory
* network_info_DeepNTax.cfg: config file containg information of model including hyperparameters
* otu_table.csv: OTU relative abundance table with samples in rows and OTUs in columns
* phenotype.csv: sample phenotype information
* taxonomic_information.csv: files containing microbiome phylogenetic informations
* deepntax: source files for the model

Then, you can run the `DeepNTax.py` script with these configuration files, for example:

```bash
nohup python3 -u DeepNTax.py network_info_DeepNTax.cfg path_info_DeepNTax.cfg > ./DeepNTax_status.txt &
```

This command will start the script in the background and redirect the output to a file named `DeepNTax_status.txt`. The `nohup` command is used to allow the script to continue running even if the terminal is closed.
