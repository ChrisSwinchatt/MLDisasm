#!/usr/bin/env python3

'''
Test mldisasm.training
'''

from mldisasm.training     import kfolds, parameter_grid
from mldisasm.tests.common import *

# Disable checking for "instance ... has no ... member" because PyTest test case instance attributes have to be defined
# outside of __init__.
# pylint: disable=E1101

class TestTraining(GenericTestCase):
    '''
    Test codecs.
    '''
    def _test_parameter_grid_iter(self):
        params = random_dict(5)
        grids  = parameter_grid(params)
        # Check that the number of grids is correct.
        size = np.product([len(params[k]) for k in params])
        assert len(grids) == size
        seen = set()
        for grid in grids:
            # Check that all the keys and values in grid are in params.
            for key, value in grid.items():
                assert key in params
                assert value in params[key]
                seen.add(value)
            # Check that no keys were left out.
            for key, value in params.items():
                assert key in grid
        # Check that every value appears in at least one grid.
        for key in params:
            for value in seen:
                if value in params[key]:
                    seen.remove(value)
        assert not seen

    def test_parameter_grid(self):
        '''
        Test mldisasm.training.parameter_grid
        '''
        self.itertest(self.test_parameter_grid, self._test_parameter_grid_iter)

    def _test_kfolds_iter(self):
        n = random_size(min_size=SEQ_LEN*2)
        k = SEQ_LEN
        all_train = set()
        all_test  = set()
        for train, test in kfolds(n, k):
            all_train.update(train)
            all_test.update(test)
            # Check that the lengths of train and test are correct.
            assert len(test)  in (n//k,n//k + 1)
            assert len(train) == n - len(test)
            # Check that no index appears twice in train.
            assert len(train) == len(set(train))
            # Check that no index appears twice in test.
            assert len(test) == len(set(test))
            # Check that no index appears in both train and test.
            indices = list(train) + list(test)
            assert len(indices) == len(set(indices))
            # Check that every index from 0 to n is in either train or test.
            assert sorted(indices) == list(range(n))
        # Check that train and test have both included all samples.
        assert sorted(all_train) == list(range(n))
        print(all_test)
        print(list(range(n)))
        assert sorted(all_test)  == list(range(n))

    def test_kfolds(self):
        '''
        Test mldisasm.training.kfolds
        '''
        self.itertest(self.test_kfolds, self._test_kfolds_iter)
