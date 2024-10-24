import unittest
import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.datasets import load_iris
from spac.transformations import knn_clustering


class TestKnnClustering(unittest.TestCase):
    def setUp(self):
        """
        This method is run before each test.

        It sets up a test AnnData object `syn_data` with the following attributes:

        `syn_data.obs['classes']`
            Class annotations for rows (approx. 50% of the rows are missing label):
            - label("no_label") = missing,
            - label(0) = mean 10,
            - label(1) = mean 100

        It also sets up a test AnnData object `adata` initialized with data from the sklearn iris dataset:

        `adata.obs["classes"]`
            The three classes for the iris (Iris setosa, Iris virginica and Iris versicolor)
        """

        ############
        # syn_data #
        ############
        n_rows = 1000

        # Generate 1000 rows, half with mean (10, 10) and half with mean (100, 100)
        mean_10 = np.random.normal(loc=10, scale=1, size=(n_rows // 2, 2))
        mean_100 = np.random.normal(loc=100, scale=1, size=(n_rows // 2, 2))
        data_rows = np.vstack((mean_10, mean_100))

        # Generate class labels, label 0 = mean 10, label 1 = mean 100
        full_class_labels = np.array([0] * (n_rows // 2) + [1] * (n_rows // 2), dtype=object)
        class_labels = np.array([x for x in full_class_labels],dtype=object)
 
        # Replace ~50% of class labels with "missing" values
        mask = np.random.rand(*full_class_labels.shape) < 0.5
        class_labels[mask] = "no_label"

        # annotation to test "missing_label" parameter
        alt_class_labels = np.array([x for x in class_labels],dtype=object)
        alt_class_labels[alt_class_labels == "no_label"] = np.nan
        
        # Combine data columns with class labels
        self.syn_dataset = data_rows

        self.syn_data = AnnData(
            X=self.syn_dataset, var=pd.DataFrame(index=["gene1", "gene2"])
        )

        self.syn_data.layers["counts"] = self.syn_dataset

        self.syn_data.obsm["derived_features"] = self.syn_dataset

        self.syn_data.obs["classes"] = class_labels

        # annotation with all labels missing
        self.syn_data.obs["all_missing_classes"] = np.array(["no_label" for x in full_class_labels])
        # annotation with all labels present
        self.syn_data.obs["no_missing_classes"] = full_class_labels
        self.syn_data.obs["alt_classes"] = alt_class_labels

        # The string for column where class labels stored in obs
        self.annotation = "classes"
        self.alt_annotation = "alt_classes"
        # The layer used for knn
        self.layer = "counts"
        # The features used for syn_data
        self.syn_features = ["gene1", "gene2"]


    def test_typical_case(self):
        # This test checks if the function correctly adds 'knn' to the
        # AnnData object's obs attribute and if it correctly sets
        # 'knn_features' in the AnnData object's uns attribute.
        knn_clustering(
            adata=self.syn_data,
            features=self.syn_features,
            annotation=self.annotation,
            layer=self.layer,
        )
        self.assertIn("knn", self.syn_data.obs)
        self.assertEqual(self.syn_data.uns["knn_features"], self.syn_features)

    def test_output_annotation(self):
        # This test checks if the function correctly adds the "output_layer"
        # to the # AnnData object's obs attribute
        output_annotation_name = "my_output_annotation"
        knn_clustering(
            adata=self.syn_data,
            features=self.syn_features,
            annotation=self.annotation,
            layer=self.layer,
            output_annotation=output_annotation_name,
        )
        self.assertIn(output_annotation_name, self.syn_data.obs)

    def test_layer_none_case(self):
        # This test checks if the function works correctly when layer is None.
        knn_clustering(
            adata=self.syn_data,
            features=self.syn_features,
            annotation=self.annotation,
            layer=None,
        )
        self.assertIn("knn", self.syn_data.obs)
        self.assertEqual(self.syn_data.uns["knn_features"], self.syn_features)

    def test_invalid_k(self):
        # This test checks if the function raises a ValueError when the
        # k argument is not a positive integer.
        with self.assertRaises(ValueError):
            knn_clustering(
                adata=self.syn_data,
                features=self.syn_features,
                annotation=self.annotation,
                layer=self.layer,
                k="invalid",
            )

    def test_trivial_label(self):
        # This test checks if the data is fully labeled or missing labels for every datapoint

        # all datapoints labeled
        with self.assertRaises(ValueError):
            knn_clustering(
                adata=self.syn_data,
                features=self.syn_features,
                annotation="no_missing_classes",
                layer=self.layer,
            )

        # no datapoints labeled
        with self.assertRaises(ValueError):
            knn_clustering(
                adata=self.syn_data,
                features=self.syn_features,
                annotation="all_missing_classes",
                layer=self.layer,
            )

    def test_clustering_accuracy(self):
        knn_clustering(
            adata=self.syn_data,
            features=self.syn_features,
            annotation=self.annotation,
            layer="counts",
            k=50,
        )

        self.assertIn("knn", self.syn_data.obs)
        self.assertEqual(len(np.unique(self.syn_data.obs["knn"])), 2)

    def test_associated_features(self):
        # Run knn using the derived feature and generate two clusters
        output_annotation = "derived_knn"
        associated_table = "derived_features"
        knn_clustering(
            adata=self.syn_data,
            features=None,
            annotation=self.annotation,
            layer=None,
            k=50,
            output_annotation=output_annotation,
            associated_table=associated_table,
        )

        self.assertEqual(len(np.unique(self.syn_data.obs[output_annotation])), 2)

    def test_missing_label(self):
        # This test checks that the missing label parameter works as intended
        #first knn call with normal data
        knn_clustering(
            adata=self.syn_data,
            features=self.syn_features,
            annotation=self.annotation,
            layer="counts",
            k=50,
            output_annotation="knn_1",
            associated_table=None,
            missing_label = "no_label"
        )
        #second knn call with alt_class data
        knn_clustering(
            adata=self.syn_data,
            features=self.syn_features,
            annotation=self.alt_annotation,
            layer="counts",
            k=50,
            output_annotation="knn_2",
            associated_table=None,
            missing_label = np.nan
        )

        #assert that they produce the same final label
        self.assertTrue(all(self.syn_data.obs["knn_1"]==self.syn_data.obs["knn_2"]))

if __name__ == "__main__":
    unittest.main()
