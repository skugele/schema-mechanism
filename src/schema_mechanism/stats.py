from __future__ import annotations
from __future__ import annotations
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Iterable

import numpy as np
from scipy import stats as stats


class ItemCorrelationTest(ABC):

    @abstractmethod
    def positive_corr_statistic(self, table: Iterable) -> float:
        pass

    @abstractmethod
    def negative_corr_statistic(self, table: Iterable) -> float:
        pass

    def validate_data(self, data: Iterable) -> np.ndarray:
        """ Raises a ValueError if data cannot be interpreted as a 2x2 array of integers.

        :param data: the iterable to validate
        :return: True if valid table; False otherwise.
        """
        table = np.array(data)
        if not (table.shape == (2, 2) and np.issubdtype(table.dtype, int)):
            raise ValueError('invalid data: must be interpretable as a 2x2 array of integers')
        return table


class DrescherCorrelationTest(ItemCorrelationTest):

    def positive_corr_statistic(self, table: Iterable) -> float:
        """ Returns the part-to-part ratio Pr(A | X) : Pr(A | not X)

        Input data should be a 2x2 table of the form: [[N(A,X), N(not A,X)], [N(A,not X), N(not A,not X)]],
        where N(A,X) is the number of events that are both A AND X

        :return: the ratio as a float, or numpy.NAN if division by zero
        """
        # raises ValueError
        table = self.validate_data(table)

        try:
            n_x = np.sum(table[0, :])
            n_not_x = np.sum(table[1, :])

            n_a_and_x = table[0, 0]
            n_a_and_not_x = table[1, 0]

            if n_x == 0 or n_not_x == 0:
                return 0.0

            # calculate conditional probabilities
            pr_a_given_x = n_a_and_x / n_x
            pr_a_given_not_x = n_a_and_not_x / n_not_x

            pr_a = pr_a_given_x + pr_a_given_not_x

            if pr_a == 0:
                return 0.0

            # the part-to-part ratio Pr(A | X) : Pr(A | not X)
            ratio = pr_a_given_x / pr_a
            return ratio
        except ZeroDivisionError:
            return 0.0

    def negative_corr_statistic(self, table: Iterable) -> float:
        """ Returns the part-to-part ratio Pr(not A | X) : Pr(not A | not X)

        Input data should be a 2x2 table of the form: [[N(A,X), N(not A,X)], [N(A,not X), N(not A,not X)]],
        where N(A,X) is the number of events that are both A AND X

        :return: the ratio as a float, or numpy.NAN if division by zero
        """
        # raises ValueError
        table = self.validate_data(table)

        try:
            n_x = np.sum(table[0, :])
            n_not_x = np.sum(table[1, :])

            n_not_a_and_x = table[0, 1]
            n_not_a_and_not_x = table[1, 1]

            if n_x == 0 or n_not_x == 0:
                return 0.0

            # calculate conditional probabilities
            pr_not_a_given_x = n_not_a_and_x / n_x
            pr_not_a_given_not_x = n_not_a_and_not_x / n_not_x

            pr_not_a = pr_not_a_given_x + pr_not_a_given_not_x

            if pr_not_a == 0:
                return 0.0

            # the part-to-part ratio between Pr(not A | X) : Pr(not A | not X)
            ratio = pr_not_a_given_x / pr_not_a
            return ratio
        except ZeroDivisionError:
            return 0.0


class BarnardExactCorrelationTest(ItemCorrelationTest):

    def positive_corr_statistic(self, table: Iterable) -> float:
        """

        :param table:
        :return:
        """
        # raises ValueError
        table = self.validate_data(table)

        return 1.0 - stats.barnard_exact(table, alternative='greater').pvalue

    def negative_corr_statistic(self, table: Iterable) -> float:
        """

        :param table:
        :return:
        """
        # raises ValueError
        table = self.validate_data(table)

        return 1.0 - stats.barnard_exact(table, alternative='less').pvalue


class FisherExactCorrelationTest(ItemCorrelationTest):

    def positive_corr_statistic(self, table: Iterable) -> float:
        """

        :param table:
        :return:
        """
        # raises ValueError
        table = self.validate_data(table)

        _, p_value = stats.fisher_exact(table, alternative='greater')
        return 1.0 - p_value

    def negative_corr_statistic(self, table: Iterable) -> float:
        """

        :param table:
        :return:
        """
        # raises ValueError
        table = self.validate_data(table)

        _, p_value = stats.fisher_exact(table, alternative='less')
        return 1.0 - p_value


def positive_correlation(table: Iterable, test: ItemCorrelationTest, threshold: float) -> bool:
    return test.positive_corr_statistic(table) >= threshold


def negative_correlation(table: Iterable, test: ItemCorrelationTest, threshold: float) -> bool:
    return test.negative_corr_statistic(table) >= threshold
