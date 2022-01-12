# TODO: GlobalStatistics need to be updated from a method that triggers all ExtendedContext and ExtendedResult updates

# TODO: Need a method to determine all of the ON state items given a state (use item pool to cycle through all
# TODO: items, and keep track of the on ones)

# TODO: might be able to memoize this with a frozen set or something like that.

# TODO: Need a method that determines item relevance


# update schema-level stats (n_action_taken or n_action_not_taken) stats

# Update Procedure for Extended Result

# for each item in the extended context
# if item is ON:
# if item was already ON in previous state, do not update positive-transition statistics

# if item is OFF
# if item was already OFF in previous state, do not update negative-transition statistics


# Update procedure for Extended Context


class SchemaMemory:
    pass


class SchemaSelection:
    """
        See Drescher, 1991, section 3.4
    """
    pass

# TODO: Need a way to suppress the creation of a new spin-off schema when a new relevant item is detected, but that
# TODO: schema already exists. Seems like schema comparisons will be necessary, but maybe there is a better way. Some
# TODO: kind of a graph traversal may also be possible, where the graph contains the "family tree" of schemas


# TODO: My guess is that statistics updates should only occur for non-activated schemas that are applicable. This
# TODO: is based on the assumption that all of the probabilities within a schema are really conditioned on the context
# TODO: being satisfied, even though this fact is implicit.
