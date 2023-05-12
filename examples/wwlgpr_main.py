# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import yaml
import argparse
import pickle
import ray
import os
import time
import wwlgpr
import numpy as np
from skopt.space import Real, Integer
from wwlgpr.Utility import ClassifySpecies, writetofile, writetofolder
from wwlgpr.WWL_GPR import BayOptCv
from sklearn.model_selection import StratifiedKFold

# -----------------------------------------------------------------------------
# LOAD DATABASE
# -----------------------------------------------------------------------------

def load_db(ml_dict, task, base_path):
    
    if task == "train":
        label = "database"
    elif task == "test":
        label = "test_database"
    
    with open(base_path+ml_dict[label]["db_graphs"], "rb") as infile:
        db_graphs = np.array(pickle.load(infile))
    with open(base_path+ml_dict[label]["db_atoms"], "rb") as infile:
        db_atoms = np.array(pickle.load(infile), dtype=object)
    with open(base_path+ml_dict[label]["node_attributes"], "rb") as infile:
        node_attributes = np.array(pickle.load(infile), dtype=object)
    with open(base_path+ml_dict[label]["target_properties"], "rb") as infile:
        list_ads_energies = np.array(pickle.load(infile))
    with open(base_path+ml_dict[label]["file_names"], "rb") as infile:
        file_names = np.array(pickle.load(infile))
    return db_graphs, db_atoms, node_attributes, list_ads_energies, file_names

# -----------------------------------------------------------------------------
# 5-FOLD CROSS VALIDATION
# -----------------------------------------------------------------------------

def SCV5(ml_dict, opt_dimensions, default_para, fix_hypers, output_name):
    """task1 and task2: 5-fold cross validation for in-domain prediction 
    stratified by adsorbate
        
     Args:
         ml_dict        ([type]): ML setting
         default_para   ([type]): user defined trials
         opt_dimensions ([type]): dimensions object for skopt
         fix_hypers     ([type]): fixed hyperparameters
     """
    
    (
        db_graphs,
        db_atoms,
        node_attributes,
        list_ads_energies,
        file_names,
    ) = load_db(ml_dict=ml_dict, task="train", base_path=base_path)
    
    test_RMSEs = []
    f_times = 1
    skf = StratifiedKFold(n_splits=5, random_state=25, shuffle=True)
    
    for train_index, vali_index in skf.split(
        list_ads_energies, 
        ClassifySpecies(file_names, db_atoms),
    ):
        
        train_db_graphs = db_graphs[train_index]
        train_db_atoms = db_atoms[train_index]
        train_node_attributes = node_attributes[train_index]
        train_list_ads_energies = list_ads_energies[train_index]
        train_file_names = file_names[train_index]

        test_db_graphs = db_graphs[vali_index]
        test_db_atoms = db_atoms[vali_index]
        test_node_attributes = node_attributes[vali_index]
        test_list_ads_energies = list_ads_energies[vali_index]
        test_file_names = file_names[vali_index]

        # Initialize bayesian optimization
        bayoptcv = BayOptCv(
            classifyspecies=ClassifySpecies(train_file_names, train_db_atoms),
            num_cpus=int(ml_dict["num_cpus"]),
            db_graphs=train_db_graphs,
            db_atoms=train_db_atoms,
            node_attributes=train_node_attributes,
            y=train_list_ads_energies,
            drop_list=None,
            num_iter=int(ml_dict["num_iter"]),
            pre_data_type=ml_dict["pre_data_type"],
            filenames=train_file_names,
        )

        # Starting bayesian optimization to minimize likelihood
        res_opt = bayoptcv.BayOpt(
            opt_dimensions=opt_dimensions,
            default_para=default_para,
            fix_hypers=fix_hypers,
        )
        print("hyperparameters:", res_opt.x)

        # Prediction with the use of optimized hyperparameters
        test_RMSE, test_pre = bayoptcv.Predict(
            test_graphs=test_db_graphs,
            test_atoms=test_db_atoms,
            test_node_attributes=test_node_attributes,
            test_target=test_list_ads_energies,
            opt_hypers=dict(zip(opt_dimensions.keys(), res_opt.x)),
        )
        print(f"{f_times} fold RMSE: ", test_RMSE)
        test_RMSEs.append(test_RMSE)
        
        if f_times == 1:
            writetofolder(output_name)
        
        writetofile(
            output_name, test_file_names, test_list_ads_energies, test_pre,
        )
        
        f_times += 1
    
    print("Cross validation RMSE: ", np.mean(test_RMSEs))

# -----------------------------------------------------------------------------
# 5-FOLD CROSS VALIDATION (FIXED PARAMETERS)
# -----------------------------------------------------------------------------

def SCV5_FHP(ml_dict, fix_hypers, output_name):
    """task3: 5-fold cross validation stratified by adsorbate with fixed 
    hyperparameters

     Args:
         ml_dict    ([type]): ML setting
         fix_hypers ([type]): fixed hyperparameters
     """
    
    (
        db_graphs,
        db_atoms,
        node_attributes,
        list_ads_energies,
        file_names,
    ) = load_db(ml_dict=ml_dict, task="train", base_path=base_path)
    
    test_RMSEs = []
    f_times = 1
    skf = StratifiedKFold(n_splits=5, random_state=25, shuffle=True)
    
    for train_index, vali_index in skf.split(
        list_ads_energies,
        ClassifySpecies(file_names, db_atoms),
    ):
        train_db_graphs = db_graphs[train_index]
        train_db_atoms = db_atoms[train_index]
        train_node_attributes = node_attributes[train_index]
        train_list_ads_energies = list_ads_energies[train_index]
        train_file_names = file_names[train_index]

        test_db_graphs = db_graphs[vali_index]
        test_db_atoms = db_atoms[vali_index]
        test_node_attributes = node_attributes[vali_index]
        test_list_ads_energies = list_ads_energies[vali_index]
        test_file_names = file_names[vali_index]

        bayoptcv = BayOptCv(
            classifyspecies=ClassifySpecies(train_file_names, train_db_atoms),
            num_cpus=int(ml_dict["num_cpus"]),
            db_graphs=train_db_graphs,
            db_atoms=train_db_atoms,
            node_attributes=train_node_attributes,
            y=train_list_ads_energies,
            drop_list=None,
            num_iter=int(ml_dict["num_iter"]),
            pre_data_type=ml_dict["pre_data_type"],
            filenames=train_file_names,
        )

        test_RMSE, test_pre = bayoptcv.Predict(
            test_graphs=test_db_graphs,
            test_atoms=test_db_atoms,
            test_node_attributes=test_node_attributes,
            test_target=test_list_ads_energies,
            opt_hypers=fix_hypers,
        )
        print(f"{f_times} fold RMSE: ", test_RMSE)
        test_RMSEs.append(test_RMSE)
        
        if f_times == 1:
            writetofolder(output_name)
        
        writetofile(
            output_name, test_file_names, test_list_ads_energies, test_pre,
        )
        
        f_times += 1
    
    print("Cross validation RMSE: ", np.mean(test_RMSEs))

# -----------------------------------------------------------------------------
# EXTRAPOLATION
# -----------------------------------------------------------------------------

def Extrapolation(ml_dict, fix_hypers, output_name):
    """task4: extrapolation to CuCo alloy and new metal (Pt) when training 
    pure metal database where additionally include atomic species on Pt in 
    the database. with pre-optimized hyperparameters.

     Args:
         ml_dict      ([type]): ML setting
         fix_hypers   ([type]): fixed hyperparameters
     """
    
    (
        train_db_graphs,
        train_db_atoms,
        train_node_attributes,
        train_list_ads_energies,
        train_file_names,
    ) = load_db(ml_dict=ml_dict, task="train", base_path=base_path)
    
    (
        test_db_graphs,
        test_db_atoms,
        test_node_attributes,
        test_list_ads_energies,
        test_file_names,
    ) = load_db(ml_dict=ml_dict, task="test", base_path=base_path)

    # Initialize bayesian optimization
    bayoptcv = BayOptCv(
        classifyspecies=ClassifySpecies(train_file_names, train_db_atoms),
        num_cpus=int(ml_dict["num_cpus"]),
        db_graphs=train_db_graphs,
        db_atoms=train_db_atoms,
        node_attributes=train_node_attributes,
        y=train_list_ads_energies,
        drop_list=None,
        num_iter=int(ml_dict["num_iter"]),
        pre_data_type=ml_dict["pre_data_type"],
        filenames=train_file_names,
    )

    test_RMSE, test_pre = bayoptcv.Predict(
        test_graphs=test_db_graphs,
        test_atoms=test_db_atoms,
        test_node_attributes=test_node_attributes,
        test_target=test_list_ads_energies,
        opt_hypers=fix_hypers,
    )
    writetofolder(output_name)
    writetofile(
        output_name, test_file_names, test_list_ads_energies, test_pre,
    )
    print("extrapolation RMSE: ", test_RMSE)

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Physic-inspired Wasserstein Weisfeiler-Lehman " +
        "Graph Gaussian Process Regression"
    )
    parser.add_argument(
        "--task",
        type=str,
        help="type of ML task",
        choices=["CV5", "CV5_FHP", "Extrapolation", "CV5_simpleads"],
    )
    parser.add_argument(
        "--uuid", 
        type=str, 
        help="uuid for ray job in HPC",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="output file name",
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        help="name of the dataset", 
    )
    parser.add_argument(
        "--base_path_src", 
        help="use the original database",
        action='store_true',
        default=False,
    )
    args = parser.parse_args()

    if args.base_path_src:
        base_path = os.path.dirname(wwlgpr.__path__[0])
    else:
        base_path = os.getcwd()

    job_name = "run_{}_{}".format(
        args.dataset,
        time.strftime("%m%d-%H%M", time.localtime())
    )

    output = args.output if args.output else job_name
    output_name = f"{base_path}/results/{output}.txt"

    if args.task == "CV5":

        #! load_setting from input.yml
        print("Load ML setting from input.yml")
        with open(f"{base_path}/database/{args.dataset}/input.yml") as f:
            ml_dict = yaml.safe_load(f)

        #! initialize ray for paralleization
        ray.init(address=os.environ["ip_head"], _redis_password=args.uuid)
        print("Nodes in the Ray cluster:", ray.nodes())

        # cutoff       = Integer(name='cutoff', low = 1, high=5)
        # inner_cutoff = Integer(name='inner_cutoff', low = 1, high=3)
        inner_weight = Real(
            name="inner_weight",
            low=0,
            high=1,
            prior="uniform",
        )
        outer_weight = Real(
            name="outer_weight",
            low=0,
            high=1,
            prior="uniform",
        )
        gpr_reg = Real(
            name="regularization of gpr",
            low=1e-3,
            high=1e0,
            prior="uniform",
        )
        gpr_len = Real(
            name="lengthscale of gpr",
            low=1,
            high=100,
            prior="uniform",
        )
        edge_s_s = Real(
            name="edge weight of surface-surface",
            low=0,
            high=1,
            prior="uniform",
        )
        edge_s_a = Real(
            name="edge weight of surface-adsorbate",
            low=0,
            high=1,
            prior="uniform",
        )
        edge_a_a = Real(
            name="edge weight of adsorbate-adsorbate",
            low=0,
            high=1,
            prior="uniform",
        )

        fix_hypers = {"cutoff": 2, "inner_cutoff": 1, "gpr_sigma": 1}

        opt_dimensions = {
            "inner_weight": inner_weight,
            "outer_weight": outer_weight,
            "gpr_reg": gpr_reg,
            "gpr_len": gpr_len,
            "edge_s_s": edge_s_s,
            "edge_s_a": edge_s_a,
            "edge_a_a": edge_a_a,
        }

        default_para = [
            [
                1.0,
                0,
                0.03,
                30,
                0,
                1,
                0,
            ],
            [
                0.6,
                0.0544362754971445,
                0.00824480194221483,
                11.4733820390901,
                0,
                1,
                0.6994924119498536,
            ],
        ]

        SCV5(
            ml_dict=ml_dict,
            opt_dimensions=opt_dimensions,
            default_para=default_para,
            fix_hypers=fix_hypers,
            output_name=output_name,
        )

    if args.task == "Extrapolation":

        # Load setting from input.yml
        print("Load ML setting from input.yml")
        with open(f"{base_path}/database/{args.dataset}/input.yml") as f:
            ml_dict = yaml.safe_load(f)

        # Initialize ray for paralleization
        ray.init(address=os.environ["ip_head"], _redis_password=args.uuid)
        print("Nodes in the Ray cluster:", ray.nodes())

        fix_hypers = {
            "cutoff": 2,
            "inner_cutoff": 1,
            "inner_weight": 0.6,
            "outer_weight": 0.0167135893463353,
            "gpr_reg": 0.02682695795279726,
            "gpr_len": 22.857142857142858,
            "gpr_sigma": 1,
            "edge_s_s": 0.49642857,
            "edge_s_a": 0.4674309212968105,
            "edge_a_a": 0.49795096939871913,
        }

        Extrapolation(
            ml_dict=ml_dict,
            fix_hypers=fix_hypers,
            output_name=output_name,
        )

    if args.task == "CV5_FHP":

        # Load setting from input.yml
        print("Load ML setting from input.yml")
        with open(f"{base_path}/database/{args.dataset}/input.yml") as f:
            ml_dict = yaml.safe_load(f)

        if "ip_head" in os.environ:
            # Running on hpc cluster
            ray.init(address=os.environ["ip_head"], _redis_password=args.uuid)
            print("Nodes in the Ray cluster:", ray.nodes())
        else:
            # Running on local desktop or laptop
            ray.init(num_cpus=ml_dict["num_cpus"])
            print("Job running on {} cpus".format(ml_dict["num_cpus"]))

        fix_hypers = {
            "cutoff": 2,
            "inner_cutoff": 1,
            "inner_weight": 0.6,
            "outer_weight": 0.0544362754971445,
            "gpr_reg": 0.00824480194221483,
            "gpr_len": 11.4733820390901,
            "gpr_sigma": 1,
            "edge_s_s": 0,
            "edge_s_a": 1,
            "edge_a_a": 0.6994924119498536,
        }

        SCV5_FHP(
            ml_dict=ml_dict,
            fix_hypers=fix_hypers,
            output_name=output_name,
        )

    if args.task == "CV5_simpleads":

        # Loadsetting from input.yml
        print("Load ML setting from input.yml")
        with open(f"{base_path}/database/{args.dataset}/input.yml") as f:
            ml_dict = yaml.safe_load(f)

        # Initialize ray for paralleization
        ray.init(address=os.environ["ip_head"], _redis_password=args.uuid)
        print("Nodes in the Ray cluster:", ray.nodes())

        cutoff = Integer(name="cutoff", low=1, high=5)
        inner_cutoff = Integer(name="inner_cutoff", low=1, high=3)
        inner_weight = Real(
            name="inner_weight",
            low=0,
            high=1,
            prior="uniform",
        )
        outer_weight = Real(
            name="outer_weight",
            low=0,
            high=1,
            prior="uniform",
        )
        gpr_reg = Real(
            name="regularization of gpr",
            low=1e-3,
            high=1e0,
            prior="uniform",
        )
        gpr_len = Real(
            name="lengthscale of gpr",
            low=1,
            high=100,
            prior="uniform",
        )
        edge_s_s = Real(
            name="edge weight of surface-surface",
            low=0,
            high=1,
            prior="uniform",
        )
        edge_s_a = Real(
            name="edge weight of surface-adsorbate",
            low=0,
            high=1,
            prior="uniform",
        )
        edge_a_a = Real(
            name="edge weight of adsorbate-adsorbate",
            low=0,
            high=1,
            prior="uniform",
        )

        fix_hypers = {"gpr_sigma": 1}

        opt_dimensions = {
            "cutoff": cutoff,
            "inner_cutoff": inner_cutoff,
            "inner_weight": inner_weight,
            "outer_weight": outer_weight,
            "gpr_reg": gpr_reg,
            "gpr_len": gpr_len,
            "edge_s_s": edge_s_s,
            "edge_s_a": edge_s_a,
            "edge_a_a": edge_a_a,
        }

        default_para = [
            [
                1.0,
                1,
                0.03,
                30,
                0,
                1,
                0,
            ],
            [
                1.0,
                0,
                0.00428050586923317,
                14.5556734435333,
                0,
                1,
                0.8678710596753815,
            ],
        ]

        SCV5(
            ml_dict=ml_dict,
            opt_dimensions=opt_dimensions,
            default_para=default_para,
            fix_hypers=fix_hypers,
            output_name=output_name,
        )

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------