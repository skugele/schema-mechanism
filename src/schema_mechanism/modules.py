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
