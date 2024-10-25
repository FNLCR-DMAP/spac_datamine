import unittest
import numpy as np
import pandas as pd
from anndata import AnnData
from spac.transformations import kmeans


class TestKmeansClustering(unittest.TestCase):
    def setUp(self):
        # This method is run before each test.
        # It sets up a test case with an AnnData object, a list of features,
        # and a layer name.
        n_cells = 100 
        self.adata = AnnData(np.random.rand(n_cells, 3),
                             var=pd.DataFrame(index=['gene1',
                                                     'gene2',
                                                     'gene3']))
        self.adata.layers['counts'] = np.random.rand(100, 3)

        self.features = ['gene1', 'gene2']
        self.layer = 'counts'

        self.syn_dataset = np.array([
                    np.concatenate(
                            (
                                np.random.normal(100, 1, 500),
                                np.random.normal(10, 1, 500)
                            )
                        ),
                    np.concatenate(
                            (
                                np.random.normal(10, 1, 500),
                                np.random.normal(100, 1, 500)
                            )
                        ),
                ]).reshape(-1, 2)

        self.syn_data = AnnData(
                self.syn_dataset,
                var=pd.DataFrame(index=['gene1',
                                        'gene2'])
                )
        self.syn_data.layers['counts'] = self.syn_dataset

        self.syn_data.obsm["derived_features"] = \
            self.syn_dataset

    def test_same_cluster_assignments_with_same_seed(self):
        # Run kmeans with a specific seed
        # and store the cluster assignments
        kmeans(self.adata, self.features, self.layer, seed=42)
        first_run_clusters = self.adata.obs['kmeans'].copy()

        # Reset the kmeans annotation and run again with the same seed
        del self.adata.obs['kmeans']
        kmeans(self.adata, self.features, self.layer, seed=42)

        # Check if the cluster assignments are the same
        self.assertTrue(
            (first_run_clusters == self.adata.obs['kmeans']).all()
        )

    def test_typical_case(self):
        # This test checks if the function correctly adds 'kmeans' to the
        # AnnData object's obs attribute and if it correctly sets
        # 'kmeans_features' in the AnnData object's uns attribute.
        kmeans(self.adata, self.features, self.layer)
        self.assertIn('kmeans', self.adata.obs)
        self.assertEqual(self.adata.uns['kmeans_features'],
                         self.features)

    def test_output_annotation(self):
        # This test checks if the function correctly adds the "output_layer" 
        # to the # AnnData object's obs attribute 
        output_annotation_name = 'my_output_annotation'
        kmeans(self.adata,
                              self.features,
                              self.layer,
                              output_annotation=output_annotation_name)
        self.assertIn(output_annotation_name, self.adata.obs)

    def test_layer_none_case(self):
        # This test checks if the function works correctly when layer is None.
        kmeans(self.adata, self.features, None)
        self.assertIn('kmeans', self.adata.obs)
        self.assertEqual(self.adata.uns['kmeans_features'],
                         self.features)

    def test_invalid_k(self):
        # This test checks if the function raises a ValueError when the
        # k argument is not a positive integer.
        with self.assertRaises(ValueError):
            kmeans(self.adata, self.features, self.layer,
                                  'invalid')

    def test_clustering_accuracy(self):
        kmeans(self.syn_data,
                              self.features,
                              'counts',
                              k=50)

        self.assertIn('kmeans', self.syn_data.obs)
        self.assertEqual(
            len(np.unique(self.syn_data.obs['kmeans'])),
            2)

    def test_associated_features(self):
        # Run kmeans using the derived feature and generate two clusters
        output_annotation = 'derived_kmeans'
        associated_table = 'derived_features'
        kmeans(
            adata=self.syn_data,
            features=None,
            layer=None,
            k=50,
            seed=None,
            output_annotation=output_annotation,
            associated_table=associated_table
        )

        self.assertEqual(
            len(np.unique(self.syn_data.obs[output_annotation])),
            2)



if __name__ == '__main__':
    unittest.main()
