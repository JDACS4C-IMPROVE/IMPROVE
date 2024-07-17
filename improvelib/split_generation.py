import numpy as np
import pandas as pd


def generate_random_splits(dataset_name, number_of_entries, out_dir, n_splits=10, random_state=1, validation_size=0.1):
    groups = np.array(list(range(number_of_entries)))
    _generate_splits(dataset_name, groups, out_dir, tag='random', n_splits=n_splits,
                     random_state=random_state, validation_size=validation_size,
                     main_cv_type='shuffle', validation_split_type='shuffle')


def _generate_scaffolds(smiles):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
    processed_scaffolds = {None: -1}
    scaffold_id = 0
    scaffold_groups = []

    for smile in smiles:
        scaffold = None
        try:
            mol = Chem.MolFromSmiles(smile)
            scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except:
            pass
        if scaffold not in processed_scaffolds:
            processed_scaffolds[scaffold] = scaffold_id
            scaffold_id += 1
        scaffold_groups.append(processed_scaffolds[scaffold])
    return scaffold_groups


def generate_scaffold_splits(dataset_name, smiles, out_dir, n_splits=10, random_state=1, validation_size=0.1):
    scaffolds = _generate_scaffolds(smiles)
    _generate_splits(dataset_name, scaffolds, out_dir, tag='scaffold', n_splits=n_splits,
                     random_state=random_state, validation_size=validation_size,
                     main_cv_type='grouped', validation_split_type='shuffle')


def _generate_splits(dataset_name, groups, out_dir, tag=None, n_splits=10, random_state=1, validation_size=0.1, main_cv_type='shuffle', validation_split_type='stratified') -> None:
    from sklearn.model_selection import StratifiedKFold, GroupKFold, GroupShuffleSplit, StratifiedShuffleSplit, KFold, ShuffleSplit
    if tag is None:
        tag = main_cv_type
    X = np.array(range(len(groups)))
    groups = np.array(groups)
    test_size = 1. / n_splits
    train_size = 1 - test_size
    validation_size = validation_size / train_size

    main_cv = None
    if main_cv_type == 'stratified':
        main_cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state)
    elif main_cv_type == 'shuffle':
        main_cv = KFold(n_splits=n_splits, shuffle=True,
                        random_state=random_state)
    elif main_cv_type == 'grouped':
        main_cv = GroupKFold(n_splits=n_splits)
    else:
        raise Exception('Unknown Main CV type!')

    validation_split = None
    if validation_split_type == 'stratified':
        validation_split = StratifiedShuffleSplit(
            n_splits=1, test_size=validation_size, random_state=random_state)
    elif validation_split_type == 'shuffle':
        validation_split = ShuffleSplit(
            n_splits=1, test_size=validation_size, random_state=random_state)
    elif validation_split_type == 'grouped':
        validation_split = GroupShuffleSplit(
            n_splits=1, test_size=validation_size, random_state=random_state)
    else:
        raise Exception('Unknown Validation CV type!')

    i = 0
    for train_index_outer, test_index in main_cv.split(X, groups, groups):
        X_train = X[train_index_outer]
        groups_train = groups[train_index_outer]

        for train_index, validation_index in validation_split.split(X_train, groups_train, groups_train):
            # X contains indices of the original dataset
            train_index = X_train[train_index]
            validation_index = X_train[validation_index]
            pd.DataFrame(train_index).to_csv(
                os.path.join(out_dir, f'{dataset_name}_{tag}_train_split_{i}.tsv'), sep='\t', index=None, header=None)
            pd.DataFrame(validation_index).to_csv(
                os.path.join(out_dir, f'{dataset_name}_{tag}_val_split_{i}.tsv'), sep='\t', index=None, header=None)
            pd.DataFrame(test_index).to_csv(
                os.path.join(out_dir, f'{dataset_name}_{tag}_test_split_{i}.tsv'), sep='\t', index=None, header=None)

            break
        i += 1
