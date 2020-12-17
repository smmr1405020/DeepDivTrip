import args_kdiverse
args_kdiverse.dataset_index = 7

import kfold_dataset_generator
kfold_dataset_generator.generate_ds_kfold_parts(5)
kfold_dataset_generator.generate_train_test_data(1, 5)

import kdiverse_generator
kdiverse_generator.generate_result(True, 3)
