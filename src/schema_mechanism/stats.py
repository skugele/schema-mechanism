from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from functools import cache
from functools import lru_cache

import numpy as np
from scipy import stats as stats

CorrelationTable = tuple[int, int, int, int]


class ItemCorrelationTest(ABC):

    @classmethod
    @abstractmethod
    def positive_corr_statistic(cls, table: CorrelationTable) -> float:
        pass

    @classmethod
    @abstractmethod
    def negative_corr_statistic(cls, table: CorrelationTable) -> float:
        pass


class DrescherCorrelationTest(ItemCorrelationTest):

    @classmethod
    @cache
    def positive_corr_statistic(cls, table: CorrelationTable) -> float:
        """ Returns the part-to-part ratio Pr(A | X) : Pr(A | not X)

        Input data should be an integer tuple of the form: [[N(A,X), N(not A,X)], [N(A,not X), N(not A,not X)]],
        where N(A,X) is the number of events that are both A AND X

        :return: the ratio as a float, or numpy.NAN if division by zero
        """

        try:
            n_x = table[0] + table[1]
            n_not_x = table[2] + table[3]

            n_a_and_x = table[0]
            n_a_and_not_x = table[2]

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

    @classmethod
    @cache
    def negative_corr_statistic(cls, table: CorrelationTable) -> float:
        """ Returns the part-to-part ratio Pr(not A | X) : Pr(not A | not X)

        Input data should be an integer tuple of the form: [N(A,X), N(not A,X), N(A,not X), N(not A,not X)],
        where N(A,X) is the number of events that are both A AND X

        :return: the ratio as a float, or numpy.NAN if division by zero
        """

        try:
            n_x = table[0] + table[1]
            n_not_x = table[2] + table[3]

            n_not_a_and_x = table[1]
            n_not_a_and_not_x = table[3]

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

    @classmethod
    @cache
    def positive_corr_statistic(cls, table: CorrelationTable) -> float:
        """

        :param table:
        :return:
        """
        array = np.array([table[0:2], table[2:]])
        return 1.0 - stats.barnard_exact(array, alternative='greater').pvalue

    @classmethod
    @cache
    def negative_corr_statistic(cls, table: CorrelationTable) -> float:
        """

        :param table:
        :return:
        """
        array = np.array([table[0:2], table[2:]])
        return 1.0 - stats.barnard_exact(array, alternative='less').pvalue


class FisherExactCorrelationTest(ItemCorrelationTest):

    @classmethod
    @lru_cache(maxsize=100000)
    def positive_corr_statistic(cls, table: CorrelationTable) -> float:
        """

        :param table:
        :return:
        """
        array = np.array([table[0:2], table[2:]])
        _, p_value = stats.fisher_exact(array, alternative='greater')
        return 1.0 - p_value

    @classmethod
    @lru_cache(maxsize=100000)
    def negative_corr_statistic(cls, table: CorrelationTable) -> float:
        """

        :param table:
        :return:
        """
        array = np.array([table[0:2], table[2:]])
        _, p_value = stats.fisher_exact(array, alternative='less')
        return 1.0 - p_value


def positive_correlation(table: CorrelationTable, test: ItemCorrelationTest, threshold: float) -> bool:
    return test.positive_corr_statistic(table) >= threshold


def negative_correlation(table: CorrelationTable, test: ItemCorrelationTest, threshold: float) -> bool:
    return test.negative_corr_statistic(table) >= threshold
