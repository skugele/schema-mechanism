import logging
from enum import Enum

logger = logging.getLogger(__name__)


class SupportedFeature(Enum):
    COMPOSITE_ACTIONS = 'COMPOSITE_ACTIONS'

    # "There is an embellishment of the marginal attribution algorithm--deferring to a more specific applicable schema--
    #  that often enables the discovery of an item whose relevance has been obscured." (see Drescher,1991, pp. 75-76)
    EC_DEFER_TO_MORE_SPECIFIC_SCHEMA = 'EC_DEFER_TO_MORE_SPECIFIC_SCHEMA'

    # "[another] embellishment also reduces redundancy: when a schema's extended context simultaneously detects the
    # relevance of several items--that is, their statistics pass the significance threshold on the same trial--the most
    # specific is chosen as the one for inclusion in a spin-off from that schema." (see Drescher, 1991, p. 77)
    #
    #     Note: Requires that EC_DEFER_TO_MORE_SPECIFIC is also enabled.
    EC_MOST_SPECIFIC_ON_MULTIPLE = 'EC_MOST_SPECIFIC_ON_MULTIPLE'

    # "The machinery's sensitivity to results is amplified by an embellishment of marginal attribution: when a given
    #  schema is idle (i.e., it has not just completed an activation), the updating of its extended result data is
    #  suppressed for any state transition which is explained--meaning that the transition is predicted as the result
    #  of a reliable schema whose activation has just completed." (see Drescher, 1991, p. 73)
    ER_SUPPRESS_UPDATE_ON_EXPLAINED = 'ER_SUPPRESS_UPDATE_ON_EXPLAINED'

    # Updates to reliable schemas' extended context item stats are suppressed when this feature is enabled. The
    # rational for this is that once a schema is reliable additional context items are no longer needed. (This
    # was not a feature of Drescher's original schema mechanism.)
    EC_SUPPRESS_UPDATE_ON_RELIABLE = 'EC_SUPPRESS_UPDATE_ON_RELIABLE'

    # Supports the creation of result spin-off schemas incrementally. This was not supported in the original schema
    # mechanism because of the proliferation of composite results that result. It is allowed here to facilitate
    # comparison and experimentation.
    ER_INCREMENTAL_RESULTS = 'ER_INCREMENTAL_RESULTS'

    # Item stats updates are frozen as soon as a correlation (positive or negative) is detected. This is a performance
    # enhancement intended to reduce the number of items for which stats much be maintained.
    FREEZE_ITEM_STATS_UPDATES_ON_CORRELATION = 'FREEZE_ITEM_STATS_UPDATES_ON_CORRELATION'
