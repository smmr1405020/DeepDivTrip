import args_kdiverse
args_kdiverse.dataset_index = 9

import kfold_dataset_generator
kfold_dataset_generator.generate_ds_kfold_parts(5)
kfold_dataset_generator.generate_train_test_data(1, 5)

import kdiverse_generator

for fold in range(1,6):
    print("Fold: {}".format(fold))
    kfold_dataset_generator.generate_train_test_data(fold, 5)
    kdiverse_generator.generate_result(False,1)
    print("\n\n")
    for K in range(2,6):
        kdiverse_generator.generate_result(True,K)