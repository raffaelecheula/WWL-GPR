import numpy as np
import ray
import time
import igraph as ig
import matplotlib.pyplot as plt
from numpy.core.numeric import outer
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from skopt import gp_minimize
from functools import partial
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
from sklearn.base import TransformerMixin
from collections import defaultdict
from typing import List
from skopt.plots import plot_convergence, plot_evaluations, plot_objective
from scipy.linalg import cholesky, cho_solve, solve_triangular
from sklearn import preprocessing
from .Utility import (
    multiprocessing_WD,
    cal_node_weights,
    ConcAttributes,
    CumsumAttributes,
    fill_nan,
    DropFeatures,
)


class GraphBase:
    """
    Base class for graph-based operations.
    """
    
    def __init__(
        self,
        db_graphs,
        db_atoms,
        node_attributes,
        y,
        drop_list=None,
        num_iter=1,
        pre_data_type="Standardize",
        filenames=None,
        num_cpus=1,
    ):

        """
        Initialize the graph base with data, configurations, and processing parameters.
        
        Args:
            db_graphs: Initial graph representations of database.
            db_atoms: ASE atoms object for initial guess.
            node_attributes: Node attributes.
            y: Target properties.
            drop_list: Features to drop. Defaults to None.
            num_iter: Number of WL iterations. Defaults to 1.
            pre_data_type: Data preprocessing. Defaults to "Standardize".
            filenames: Name of structures. Defaults to None.
            num_cpus: Number of cpus. Defaults to 1.
        """

        self.db_graphs = np.asarray(db_graphs)
        self.db_atoms = np.asarray(db_atoms)
        self.node_attributes = np.asarray(node_attributes, dtype=object)
        self.y = np.asarray(y)
        self.filenames = np.asarray(filenames)

        self.drop_list = drop_list
        self.num_iter = num_iter
        self.pre_data_type = pre_data_type

        GraphBase.num_cpus = num_cpus

    @staticmethod
    def gpr_kernel_matrix(
        db_graphs,
        db_atoms,
        post_node_attributes,
        gpr_hypers_dict,
    ):
        """
        Calculate covariance matrix based on graphs and hyperparameters.

        Args:
            db_graphs: Initial graph representations of database.
            db_atoms: ASE atoms object for initial guess.
            post_node_attributes: Node features.
            gpr_hypers_dict: Dictionary with hyperparameters.

        Returns:
            Covariance matrix (NxN).
        """
        n = len(db_graphs)
        res_node_weight = np.zeros(n, dtype=object)
        for graph_index, graph in enumerate(db_graphs):
            res_node_weight[graph_index] = cal_node_weights(
                graph,
                db_atoms[graph_index],
                gpr_hypers_dict["cutoff"],
                gpr_hypers_dict["inner_cutoff"],
                gpr_hypers_dict["inner_weight"],
                gpr_hypers_dict["outer_weight"],
            )
        post_node_attributes = ray.put(post_node_attributes) # Node features.
        node_weights = ray.put(res_node_weight) # Node weights.
        M = np.zeros((n, n))
        triu = np.triu_indices(n)
        graph_pair_index = np.asarray([[i, j] for i in range(n) for j in range(n-i)])
        splited_graph = np.array_split(graph_pair_index, GraphBase.num_cpus)
        result_ids = [
            multiprocessing_WD.remote(
                ray.put(graph_index), post_node_attributes, node_weights
            )
            for graph_index in splited_graph
        ]
        res = np.concatenate(ray.get(result_ids))
        M[triu] = res.ravel()
        M = M + M.T
        M = gpr_hypers_dict["gpr_sigma"] ** 2 * linear_kernel(
            M / gpr_hypers_dict["gpr_len"]
        )
        return M


class BayOptCv(GraphBase):
    """
    Class for Bayesian Optimization with cross-validation.
    """

    # It will be optimized during Bayesian optimization.
    default_gpr_hyper = {
        "cutoff": 2,
        "inner_cutoff": 1,
        "inner_weight": 0.60,
        "outer_weight": 0.05,
        "gpr_reg": 0.008,
        "gpr_len": 11.5,
        "gpr_sigma": 1,
        "edge_s_s": 0,
        "edge_s_a": 1,
        "edge_a_a": 0.7,
    }

    def __init__(
        self,
        classifyspecies=None,
        *args,
        **kwgs,
    ):
        """
        Initialize with optional species classification.
        """
        super().__init__(*args, **kwgs)

        self.classifyspecies = classifyspecies
        self.train = False

        if self.pre_data_type == "Normalize":
            self.data_processor = preprocessing.MaxAbsScaler()
        elif self.pre_data_type == "Standardize":
            self.data_processor = preprocessing.StandardScaler()
        else:
            raise TypeError("Unknown preprocessing type")

    def _update_default_hypers(
        self,
        hyperpars,
        name_hypers,
        fix_hypers,
    ):
        """
        Update hyperparameters with the provided values.
        """
        input_hyper_dict = dict(zip(name_hypers, hyperpars))
        BayOptCv.default_gpr_hyper.update(fix_hypers)
        BayOptCv.default_gpr_hyper.update(input_hyper_dict)

    def Preprocessing_NodeAttr(
        self,
        drop_list,
        train_node_attributes,
        test_node_attributes=None,
    ):
        """
        Preprocess node attributes for training and testing datasets.
        """
        if self.train == False:

            if drop_list is not None:
                train_node_attributes = DropFeatures(train_node_attributes, drop_list)
            train_conc_attributes = fill_nan(ConcAttributes(train_node_attributes))

            self.data_processor.fit(train_conc_attributes)
            pre_conc_attributes = self.data_processor.transform(train_conc_attributes)

            indexlist = np.cumsum(list(map(
                lambda node_attribute: len(node_attribute), train_node_attributes,
            )))
            preprocess_node_attributes = CumsumAttributes(
                pre_conc_attributes, indexlist
            )

        if self.train == True and test_node_attributes is not None:

            if drop_list is not None:
                train_node_attributes = DropFeatures(train_node_attributes, drop_list)
                test_node_attributes = DropFeatures(test_node_attributes, drop_list)

            train_conc_attributes = fill_nan(ConcAttributes(train_node_attributes))
            test_conc_attributes = fill_nan(ConcAttributes(test_node_attributes))

            # Fit model on training, transform to training and test individually.
            self.data_processor.fit(train_conc_attributes)
            train_pre_conc_attributes = self.data_processor.transform(
                train_conc_attributes
            )
            test_pre_conc_attributes = self.data_processor.transform(
                test_conc_attributes
            )
            pre_conc_attributes = np.concatenate(
                (train_pre_conc_attributes, test_pre_conc_attributes)
            )

            indexlist = np.cumsum(
                list(map(
                    lambda node_attribute: len(node_attribute), train_node_attributes,
                ))
                + list(map(
                    lambda node_attribute: len(node_attribute), test_node_attributes,
                ))
            )
            preprocess_node_attributes = CumsumAttributes(
                pre_conc_attributes, indexlist
            )

        return preprocess_node_attributes

    def _Predict_prepare(
        self,
        test_graphs,
        test_atoms,
        test_node_attributes,
        gpr_hypers_dict,
    ):
        """
        Prepare data for prediction.
       
        Args:
            test_graphs: Graphs of test data.
            test_atoms: ASE atoms of test data.
            test_node_attributes: Node attributes of test data.
            gpr_hypers_dict: Hyperparameters dictionary.

        Returns:
        Concatenated graph representations (training + test).
        Concatenated atoms object (training + test).
        """
        assert len(test_node_attributes) > 1
        conc_db_graphs = np.append(self.db_graphs, test_graphs)
        conc_db_atoms = np.append(self.db_atoms, test_atoms)
        preprocessing_node_attributes = self.Preprocessing_NodeAttr(
            self.drop_list, self.node_attributes, test_node_attributes
        )
        self.post_node_attributes = R_conv_attributes(
            conc_db_graphs,
            preprocessing_node_attributes,
            num_iterations=self.num_iter,
            gpr_hypers_dict=gpr_hypers_dict,
            db_atoms=conc_db_atoms,
        )
        return conc_db_graphs, conc_db_atoms

    def Predict(
        self,
        test_graphs,
        test_atoms,
        test_node_attributes,
        test_target,
        opt_hypers,
    ):
        """
        Perform prediction on the test data.

        Args:
            test_graphs: Graphs of test data.
            test_atoms: ASE atoms of test data.
            test_node_attributes: Node attributes of test data.
            test_target: Target property of test data.
            opt_hypers: Optimized hyperparameters.

        Returns:
            Root mean squared error (RMSE).
            Predictions.
        """
        self.train = True
        BayOptCv.default_gpr_hyper.update(opt_hypers)
        conc_db_graphs, conc_db_atoms = self._Predict_prepare(
            test_graphs,
            test_atoms,
            test_node_attributes,
            gpr_hypers_dict=BayOptCv.default_gpr_hyper,
        )
        total_n = len(conc_db_graphs)
        conc_kernel_matrix = self.__class__.gpr_kernel_matrix(
            db_graphs=conc_db_graphs,
            db_atoms=conc_db_atoms,
            post_node_attributes=self.post_node_attributes,
            gpr_hypers_dict=BayOptCv.default_gpr_hyper,
        )
        total_diag = np.zeros((total_n, total_n))
        np.fill_diagonal(
            total_diag,
            [BayOptCv.default_gpr_hyper["gpr_reg"]] * int(total_n - len(test_target))
            + [0] * len(test_target),
        )
        conc_kernel_matrix = conc_kernel_matrix + total_diag
        train_matrix = (
            conc_kernel_matrix[:len(self.y), :len(self.y)] + 1e-8*np.eye(len(self.y))
        )
        test_matrix = conc_kernel_matrix[len(self.y):, :len(self.y)]
        Kfy = conc_kernel_matrix[:len(self.y), len(self.y):]
        Kff_inv = np.linalg.inv(train_matrix + 1e-8*np.eye(len(self.y)))
        mu = Kfy.T.dot(Kff_inv).dot(self.y)
        test_RMSE = mean_squared_error(mu, test_target, squared=False)
        
        return test_RMSE, mu

    def _LossFunc(
        self,
        hyperpars,
        name_hypers,
        fix_hypers,
        preprocess_node_attributes,
    ):
        """
        Calculate the loss function (negative log marginal likelihood).

        Args:
            hyperpars: Values of hyperparameters.
            name_hypers: Name of hyperparameters to optimize.
            fix_hypers: Fixed hyperparameters.
            preprocess_node_attributes: Node attributes after preprocessing.

        Returns:
            Negative log marginal likelihood.
        """

        self._update_default_hypers(hyperpars, name_hypers, fix_hypers)
        if (
            "edge_s_s" not in name_hypers
            and "edge_s_a" not in name_hypers
            and "edge_a_a" not in name_hypers
        ):
            self.post_node_attributes = R_conv_attributes(
                self.db_graphs,
                preprocess_node_attributes,
                num_iterations=self.num_iter,
                gpr_hypers_dict=self.default_gpr_hyper,
                db_atoms=self.db_atoms,
            )
            self.edge_weights = False
        else:
            self.edge_weights = True

        if self.edge_weights == True:
            self.post_node_attributes = R_conv_attributes(
                self.db_graphs,
                preprocess_node_attributes,
                num_iterations=self.num_iter,
                gpr_hypers_dict=self.default_gpr_hyper,
                db_atoms=self.db_atoms,
            )

        pre_kernel = self.__class__.gpr_kernel_matrix(
            db_graphs=self.db_graphs,
            db_atoms=self.db_atoms,
            post_node_attributes=self.post_node_attributes,
            gpr_hypers_dict=self.default_gpr_hyper,
        )

        assert self.train == False # For testing.
        pre_kernel = pre_kernel+self.default_gpr_hyper["gpr_reg"]*np.eye(len(self.y))
        Kxx = pre_kernel
        train_y = self.y[:, np.newaxis]
        L = cholesky(Kxx, lower=True)
        alpha = cho_solve((L, True), train_y)
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", train_y, alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= Kxx.shape[0] / 2 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(-1) # Sum over dimensions.
        print("loss:", -log_likelihood)
        return -log_likelihood

    def BayOpt(
        self,
        opt_dimensions,
        default_para,
        fix_hypers,
        checkpoint_saver=None,
        n_calls=3,
    ):
        """
        Perform Bayesian Optimization using `skopt`.

        Args:
            opt_dimensions: Dimensions object of skopt to optimize.
            default_para: User defined parameters.
            fix_hypers: Fixed hyperparameters.
            checkpoint_saver: [description]. Defaults to None.

        Returns:
            Log likelihood for all tested hyperparameters
        """

        name_hypers = list(opt_dimensions.keys())
        dimensions = list(opt_dimensions.values())

        assert self.train == False
        preprocess_node_attributes = self.Preprocessing_NodeAttr(
            self.drop_list, self.node_attributes, test_node_attributes=None
        )

        if len(fix_hypers) == 10 and len(name_hypers) == 0:
            print("Fix all hyperparameters")
            self._LossFunc(
                hyperpars=dimensions,
                name_hypers=name_hypers,
                fix_hypers=fix_hypers,
                preprocess_node_attributes=preprocess_node_attributes,
            )
        else:
            res = gp_minimize(
                func=partial(
                    self._LossFunc,
                    name_hypers=name_hypers,
                    fix_hypers=fix_hypers,
                    preprocess_node_attributes=preprocess_node_attributes,
                ),
                dimensions=dimensions,
                n_calls=n_calls,
                n_random_starts=1,
                acq_func="EI",
                x0=default_para,
                xi=0.01,
                # callback = [checkpoint_saver],
                random_state=0,
            )
            # plot_convergence(res)
            # plt.savefig('convergence.png'); plt.close()
            return res
        self.train = True


class ContinuousWeisfeilerLehman(TransformerMixin):
    """
    Class that implements the continuous Weisfeiler-Lehman propagation scheme.
    """

    def __init__(self):
        self._results = {} # defaultdict(dict)
        self._label_sequences = []

    def _preprocess_graphs(self, X: List[ig.Graph]):
        """
        Preprocess graphs and extract node features and adjacency matrices.
        """
        # Initialize.
        node_features = []
        adj_mat = []
        n_nodes = []
        # Iterate across graphs and load initial node features.
        for graph in X:
            if not "label" in graph.vs.attribute_names():
                graph.vs["label"] = list(map(str, [l for l in graph.vs.degree()]))
            # Get features and adjacency matrix
            node_features_cur = graph.vs["label"]
            adj_mat_cur = np.asarray(graph.get_adjacency().data)
            # Load features
            node_features.append(
                np.asarray(node_features_cur).astype(float).reshape(-1, 1)
            )
            adj_mat.append(adj_mat_cur.astype(int))
            n_nodes.append(adj_mat_cur.shape[0])

        # By default, keep degree or label as features, if other features shall
        # to be used (e.g. the one from the TU Dortmund website),
        # provide them to the fit_transform function.
        n_nodes = np.asarray(n_nodes)
        node_features = np.asarray(node_features, dtype=object)

        return node_features, adj_mat, n_nodes

    def _create_adj_avg(self, adj_cur):
        """
        Create average adjacency matrix.
        """
        deg = np.sum(adj_cur, axis=1)
        deg = np.asarray(deg).reshape(-1)

        deg[deg != 1] -= 1

        deg = 1 / deg
        deg_mat = np.diag(deg)
        adj_cur = adj_cur.dot(deg_mat.T).T
        return adj_cur

    def _create_adj_weighted(self, graph_i, db_atom_i, gpr_hypers_dict):
        """
        Create weighted adjacency matrix.
        """
        edge_list = graph_i.get_edgelist()
        weights = np.zeros((len(db_atom_i), len(db_atom_i)))
        if 'indices_ads' in db_atom_i.info:
            indices_ads = db_atom_i.info['indices_ads']
            for edge_i in edge_list:
                matches = [ii for ii in edge_i if ii in indices_ads]
                if len(matches) == 0:
                    weights[edge_i] = gpr_hypers_dict["edge_s_s"]
                elif len(matches) == 1:
                    weights[edge_i] = gpr_hypers_dict["edge_s_a"]
                elif len(matches) == 2:
                    weights[edge_i] = gpr_hypers_dict["edge_a_a"]
                else:
                    raise TypeError("Unknown type of edge")
        else:
            for edge_i in edge_list:
                node_numbers = db_atom_i.numbers[np.array(edge_i)]
                if sum(node_numbers > 18) == 2:
                    weights[edge_i] = gpr_hypers_dict["edge_s_s"]
                elif sum(node_numbers > 18) == 1:
                    weights[edge_i] = gpr_hypers_dict["edge_s_a"]
                elif sum(node_numbers > 18) == 0:
                    weights[edge_i] = gpr_hypers_dict["edge_a_a"]
                else:
                    raise TypeError("Unknown type of edge")
        weights_sum = weights + weights.T
        degree = np.sum(weights_sum, axis=0).reshape(-1, 1)
        edge_weights_mat = np.divide(weights_sum, degree, where=degree != 0)
        return edge_weights_mat

    def fit_transform(
        self,
        X: List[ig.Graph],
        node_features=None,
        num_iterations: int = 3,
        gpr_hypers_dict=None,
        db_atoms=None,
    ):
        """
        Transform a list of graphs into their node representations. 
        Node features should be provided as a numpy array.
        """
        node_features_labels, adj_mat, n_nodes = self._preprocess_graphs(X)
        if node_features is None:
            node_features = node_features_labels
        node_features = node_features
        n_graphs = len(node_features)
        self._label_sequences = []
        if (
            gpr_hypers_dict["edge_s_s"] is None
            and gpr_hypers_dict["edge_s_a"] is None
            and gpr_hypers_dict["edge_a_a"] is None
        ):
            for i in range(n_graphs):
                graph_feat = []
                for it in range(num_iterations + 1):
                    if it == 0:
                        graph_feat.append(node_features[i])
                    else:
                        adj_cur = adj_mat[i] + np.identity(adj_mat[i].shape[0])
                        adj_cur = self._create_adj_avg(adj_cur)
                        np.fill_diagonal(adj_cur, 0)
                        graph_feat_cur = 0.5 * (
                            np.dot(adj_cur, graph_feat[it - 1]) + graph_feat[it - 1]
                        )
                        graph_feat.append(graph_feat_cur)

                self._label_sequences.append(np.concatenate(graph_feat, axis=1))
            return self._label_sequences
        else:
            for i in range(n_graphs):
                graph_feat = []
                for it in range(num_iterations + 1):
                    if it == 0:
                        graph_feat.append(node_features[i])
                    else:
                        edge_weights_mat = self._create_adj_weighted(
                            X[i], db_atoms[i], gpr_hypers_dict
                        )
                        graph_feat_cur = 0.5 * (
                            np.dot(edge_weights_mat, graph_feat[it - 1])
                            + graph_feat[it - 1]
                        )
                        graph_feat.append(graph_feat_cur)
                self._label_sequences.append(np.concatenate(graph_feat, axis=1))
            return self._label_sequences


def R_conv_attributes(
    X,
    node_features=None,
    num_iterations=3,
    gpr_hypers_dict=None,
    db_atoms=None,
    enforce_continuous=False,
):
    """
    Pairwise computation of the Wasserstein distance between embeddings of the
    graphs in X.
    
    args:
        X: List of graphs.
        node_features: Array containing the node features for continuous
            attributed graphs.
        num_iterations: Number of iterations for the propagation scheme.
    """
    # First check if the graphs are continuous vs categorical.
    categorical = True
    if enforce_continuous:
        # print('Enforce continous flag is on, using CONTINUOUS propagation scheme.')
        categorical = False
    elif node_features is not None:
        # print('Continuous node features provided, using CONTINUOUS propagation scheme.')
        categorical = False
    else:
        for g in X:
            if not "label" in g.vs.attribute_names():
                print(
                    "No label attributed to graphs, use degree instead and " +
                    "use CONTINUOUS propagation scheme."
                )
                categorical = False
                break
        if categorical:
            print(
                "Categorically-labelled graphs, using CATEGORICAL propagation scheme."
            )
    # Embed the nodes
    if categorical:
        pass
    else:
        es = ContinuousWeisfeilerLehman()
        node_representations = es.fit_transform(
            X,
            node_features=node_features,
            num_iterations=num_iterations,
            gpr_hypers_dict=gpr_hypers_dict,
            db_atoms=db_atoms,
        )
    return node_representations
