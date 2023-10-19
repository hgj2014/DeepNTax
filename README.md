# DeepNTax

DeepNTax is a DNN model that innovatively incorporates taxonomic rank information into its regularization terms. Taxonomic rank information was also used in defining the structure of the model. Taxonomic rank is used in DeepNTax for two reasons. First, the taxonomic rank structure of the data can be directly represented with each layer of the model. Secondly, using taxonomic rank rather than a phylogenetic tree makes regularization terms explicit and less affected by the misspecifi-cation of the phylogenetic tree. The taxonomic information was encompassed in regularization for the proposed model. 

In DeepNTax, the regularization matrix is constructed using the two ideas from the taxonomic information. First, closer taxonomic information between taxa could indicate that they share similar biologi-cal traits. Second, taxa that differ at a higher taxonomic rank could represent a relatively larger difference between the taxa. 
