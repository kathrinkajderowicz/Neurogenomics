import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scanpy as sc
import sklearn
from anndata import read_h5ad
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

parameters = {"myeloid" : True, 
              "non_myeloid" : True,
              
              "min_genes" : 250,
              "max_genes" : 6000, #??
              "min_transcripts" : 200,
              "max_transcripts" : 1e7,
              "min_cells" : 3, 
              
              "highly_variable_genes" : True,
              "target_sum" : 1e4,
              "min_mean" : 0.0125,
              "max_mean" : 3,
              "min_disp" : 0.2,
              
              "highly_varied_genes" : False,
              "keep_percent_genes" : 0.5,
              
              "cell_classes" : None, # None for all or choose from ["neuron", "macrophage", "endothelial cell", "brain pericyte", "astrocyte", "microglial cell", "oligodendrocyte"]
              "eliminate_18m" : True, 
              
              "mutual_information": False,
              "mutual_info_num" : 0.05,
              "mutual_info_path" : "aging_genes",
              
              "train_test_split" : 0.3,
              
              "compute_raw" : False,
              
              "compute_normalize" : False, # Set False if you want PCA!
              
              "compute_PCA" : True,
              "num_PCA_components" : 10,
             }
             
def scRNA_dataset(dir_, **kwargs):
    
    print("Loading the data...")
    
    brain = sc.read_h5ad(dir_)
    brain.var_names_make_unique()
    pd.set_option('display.max_rows', None)
    
    if kwargs["myeloid"] and kwargs["non_myeloid"]:
        brain_datasets = ["Brain_Non-Myeloid", "Brain_Myeloid"]
    elif kwargs["myeloid"]:
        brain_datasets = ["Brain_Myeloid"]
    elif kwargs["non_myeloid"]:
        brain_datasets = ["Brain_Non-Myeloid"]
    else:
        print("You need to select True for myeloid and/or non_myeloid...")
        return None, None, None, None
    
    brain_data = brain[brain.obs.tissue.isin(brain_datasets)]
    
    print("Data loaded (dimension:", brain_data.shape, ")!")
    
    print("Removing cells and genes...")

    # Remove cells with too many or too little expressed genes
    brain_data = brain_data[brain_data.obs.n_genes > kwargs["min_genes"], :]
    brain_data = brain_data[brain_data.obs.n_genes < kwargs["max_genes"], :]
    
    # Remove cells with too many or too little transcript genes
    brain_data = brain_data[brain_data.obs.n_counts < kwargs["max_transcripts"], :]
    brain_data = brain_data[brain_data.obs.n_counts > kwargs["min_transcripts"], :]
    
    # Remove genes only present in less than Y % of cells
    sc.pp.filter_genes(brain_data, min_cells = kwargs["min_cells"]*brain_data.X.shape[0]) 
    
    if (kwargs["highly_variable_genes"] and kwargs["highly_varied_genes"]) or (not kwargs["highly_variable_genes"] and not kwargs["highly_varied_genes"]):
        print("You need to select True for highly_expressed_genes or highly_varied_genes...")
        return None, None, None, None
    
    elif kwargs["highly_variable_genes"]:
        print("Highly variable gene selection...")
        brain_data2 = brain_data.copy()
        
        # Total-count normalize the data matrix X to 10,000 reads per cell - counts are comparable among cells
        sc.pp.normalize_total(brain_data2, target_sum = kwargs["target_sum"])

        # Logarithmize the data
        sc.pp.log1p(brain_data2) 

        # Identify highly variable genes 
        sc.pp.highly_variable_genes(brain_data2, min_mean = kwargs["min_mean"], max_mean = kwargs["max_mean"], 
                                    min_disp = kwargs["min_disp"])

        # Filter to keep highly variable
        brain_data = brain_data[:, brain_data2.var.highly_variable] 

    elif kwargs["highly_varied_genes"]:
        # Compute normalized variance of data and keep the max n_top_genes 
        print("Highly varied gene selection...")
        sc.pp.highly_variable_genes(brain_data, flavor = "seurat_v3", n_top_genes = int(brain_data.var_names.shape[0]*kwargs["keep_percent_genes"]))
    
    brain_data_pd = pd.DataFrame(brain_data.X.toarray(), columns = brain_data.var.index.tolist(), index = brain_data.obs.cell.tolist())
    age_label_pd = brain_data.obs.age
    
    # Selecting specific cell types
    cell_classes = brain_data.obs.cell_ontology_class.unique()
    if kwargs["cell_classes"]:
        
        print("Selecting different cell types...")
        for cell_class in kwargs["cell_classes"]:
            if cell_class not in cell_classes:
                print("This cell type was not found:", cell_class)
                print("Choose from:", cell_classes)
                return None, None, None, None
        if len(kwargs["cell_classes"]) > 1:
            cell_indxs = np.logical_or([np.array(brain_data.obs.cell_ontology_class == name) for name in kwargs["cell_classes"]])
        else:
            cell_indxs = np.array(brain_data.obs.cell_ontology_class == kwargs["cell_classes"][0])
            
        brain_data_pd = brain_data_pd[cell_indxs]
        age_label_pd = age_label_pd[cell_indxs]
        
    # More dimension reduction via gene selection
    if kwargs["mutual_information"]:
        print("Running mutual information...")
        try:
            gene_names = np.load(kwargs["mutual_info_path"])
            important_genes = []
            
            for gene in brain_data.var_names:
                if gene in gene_names:
                    important_genes.append(True)
                else:
                    important_genes.append(False)
            
            brain_data_pd = brain_data_pd[brain_data_pd.columns[important_genes]]
                    
        except:
            brain_data_norm = (brain_data_pd - brain_data_pd.mean()) / brain_data_pd.std()
            mutual_info = MIC(brain_data_norm, age_label_pd)

            important_genes = mutual_info>kwargs["mutual_info_num"]
            gene_names = brain_data.var_names[important_genes]
            
            np.save(kwargs["mutual_info_path"], gene_names)
            
            brain_data_pd = brain_data_pd[brain_data_pd.columns[important_genes]]
        
    # Returning the training testing
    X_train, X_test, y_train, y_test = train_test_split(brain_data_pd, age_label_pd, test_size=kwargs["train_test_split"], random_state=42)
    
    if kwargs["compute_raw"]:  
        print("Returning raw data!")
        return X_train, X_test, y_train, y_test
    else:
        X_train, X_test  = X_train.T, X_test.T
        X_train = (X_train - X_train.mean()) / X_train.std()
        X_test = (X_test - X_test.mean()) / X_test.std()       
        X_train, X_test  = X_train.T, X_test.T
        
        if kwargs["compute_normalize"]:  
            print("Returning normalized data!")
            return X_train, X_test, y_train, y_test

        elif kwargs["compute_PCA"]: 
            print("Returning PCA components!")
            pca = PCA(n_components=kwargs["num_PCA_components"])
            pca.fit(X_train)
            columns = ['pca_%i' % i for i in range(1, kwargs["num_PCA_components"]+1)]
            X_train = pd.DataFrame(pca.transform(X_train), columns=columns, index=X_train.index)
            X_test = pd.DataFrame(pca.transform(X_test), columns=columns, index=X_test.index)

            return X_train, X_test, y_train, y_test

        else:
            print("Returning normalized data (woot)!")
            return X_train, X_test, y_train, y_test
        
        
def load_tabula_muris(dir_, **kwargs):
    
    print("Loading the data...")
    
    brain = sc.read_h5ad(dir_)
    brain.var_names_make_unique()
    pd.set_option('display.max_rows', None)
    
    if kwargs["myeloid"] and kwargs["non_myeloid"]:
        brain_datasets = ["Brain_Non-Myeloid", "Brain_Myeloid"]
    elif kwargs["myeloid"]:
        brain_datasets = ["Brain_Myeloid"]
    elif kwargs["non_myeloid"]:
        brain_datasets = ["Brain_Non-Myeloid"]
    else:
        print("You need to select True for myeloid and/or non_myeloid...")
        return None, None, None, None
    
    brain_data = brain[brain.obs.tissue.isin(brain_datasets)]
    
    if kwargs["cell_classes"]:
        brain_data = brain_data[brain_data.obs['cell_ontology_class'] == kwargs["cell_classes"]]
    if kwargs['eliminate_18m']:
        brain_data = brain_data[brain_data.obs['age'] != '18m']
        
    print("Data loaded (dimension:", brain_data.shape, ")!")
    
    print("Removing cells and genes...")

    # Remove cells with too many or too little expressed genes
    brain_data = brain_data[brain_data.obs.n_genes > kwargs["min_genes"], :]
    brain_data = brain_data[brain_data.obs.n_genes < kwargs["max_genes"], :]
    
    # Remove cells with too many or too little transcript genes
    brain_data = brain_data[brain_data.obs.n_counts < kwargs["max_transcripts"], :]
    brain_data = brain_data[brain_data.obs.n_counts > kwargs["min_transcripts"], :]
    
    # Remove genes only present in less than Y % of cells
    sc.pp.filter_genes(brain_data, min_cells = kwargs["min_cells"]) 
    
    print("Highly variable gene selection...")

    # Total-count normalize the data matrix X to 10,000 reads per cell - counts are comparable among cells
    sc.pp.normalize_total(brain_data, target_sum = kwargs["target_sum"])

    # Logarithmize the data
    sc.pp.log1p(brain_data) 

    brain_data.raw = brain_data
    
    # Identify highly variable genes 
    sc.pp.highly_variable_genes(brain_data, min_mean = kwargs["min_mean"], max_mean = kwargs["max_mean"], 
                                min_disp = kwargs["min_disp"])

    # Filter to keep highly variable
    brain_data = brain_data[:, brain_data.var.highly_variable] 
    
    print("Data final dimension:", brain_data.shape)

    return brain_data


