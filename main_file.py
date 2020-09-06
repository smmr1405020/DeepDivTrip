import args_kdiverse
args_kdiverse.dataset_index = 2

import kfold_dataset_generator
import kdiverse_generator

kfold_dataset_generator.generate_ds_kfold_parts(5)

for fold in range(1,6):
    kfold_dataset_generator.generate_train_test_data(fold, 5)
    kdiverse_generator.generate_result(False,1)
    print("\n\n")
    for K in range(2,6):
        kdiverse_generator.generate_result(True,K)

