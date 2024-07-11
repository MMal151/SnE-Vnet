# SnE-Vnet

The configuration file can be used to either train or test the model. Currently, we have support for executing baseline models (Unet and VNet) and the presented model (Vnet-Sne).

Execute main.py to start the process. The configurations will be read from the config.yml file (a breakdown of each configuration is present within the file.). 
For the model-based configurations, refer to ConfigurationFiles/ModelConfigurations.yml. 

## Dataset Preparation
The dataset for input is expected to be in this format. The project expects the input files (T1 files) and lesion masks to have the same naming format. An example of the folder structure used is given below:

Folder Structure:
```
|____data
| |____Set_1
| | |____sub_001
| | | |____sub-r001_LESION.nii.gz
| | | |____sub-r001_T1.nii.gz
| | |____sub_002
| | | |____sub-r002_LESION.nii.gz
| | | |____sub-r002_T1.nii.gz
...
| |____Set_2
| | |____sub_003
| | | |____sub-r003_LESION.nii.gz
| | | |____sub-r003_T1.nii.gz
...

```

The suffix for lesion masks and input masks can be configured in ConfigurationsFiles/DataPrepConfig.yml.
