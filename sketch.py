class Item:
    INITIAL_GENERALITY = 0.0
    INITIAL_ACCESSIBILITY = 0.0
    INITIAL_PRIMITIVE_VALUE = 0.0
    INITIAL_DELEGATED_VALUE = 0.0
    INITIAL_INSTRUMENTAL_VALUE = 0.0

    def __init__(self, state):
        # TODO: What are the consequences of changing the state to a non-binary value?
        self.state = state

        # TODO: May of these properties will depend on other values. It's likely that
        # TODO: the properties will need to be calculated via a @property method call from
        # TODO: these other values.

        self.generality = Item.INITIAL_GENERALITY
        self.accessibility = Item.INITIAL_ACCESSIBILITY

        # “The schema mechanism explicitly designates an item as corresponding to a top-level goal by assigning the
        # item a positive value; an item can also take on a negative value, indicating a state to be avoided.”
        # (see Drescher, 1991, p. 61)
        # (see also, Drescher, 1991, Section 3.4.1)
        self.primitive_value = Item.INITIAL_PRIMITIVE_VALUE
        self.delegated_value = Item.INITIAL_DELEGATED_VALUE

        # TODO: This may not be needed.
        self.instrumental_value = Item.INITIAL_INSTRUMENTAL_VALUE

        # ratio of the probability of the slot’s item turning On when the schema’s action has just been taken to the
        # probability of its turning On when the schema’s action is not being taken
        # TODO: Does this update occur even when the context has not been satisfied??? I am going to assume that
        # TODO: updates only occur when the associated schema is activated (either explicit or implicit)
        self.positive_transition_correlation = None

        # ratio of the probability of the slot’s item turning Off when the schema’s action has just been taken to the
        # probability of its turning Off when the schema’s action is not being taken
        # TODO: Does this update occur even when the context has not been satisfied??? I am going to assume that
        # TODO: updates only occur when the associated schema is activated (either explicit or implicit)
        self.negative_transition_correlation = None

        # NOTE: “a trial for which the result was already satisfied before the action was taken does not count as
        # a positive-transition trail; and one for which the result was already unsatisfied does not count as a
        # negative-transition trial” (see Drescher, 1991, p72)

    def is_on(self, current_state):
        pass

    def is_off(self, current_state):
        pass

    def update_statistics(self, ):
        pass


# TODO: may need to split contexts and results into sensory and perceptual states
# TODO: how is this distinction handled in Drescher's implementation? Are primitive items
# TODO: considered "sensory" and synthetic items considered "conceptual"?


class Action:
    def __init__(self):
        # A unique identifier for this action
        self.id = get_unique_id()

    def __eq__(self, other):
        if self is other:
            return True
        if isinstance(other, Action):
            return self.id == other.id
        return NotImplemented

    def __ne__(self, other):
        equal = self.__eq__(other)
        if equal is NotImplemented:
            return NotImplemented
        return not equal


class PrimitiveAction(Action):
    """
        Each primitive action identifies a motor controller hard-wired to a device that carries out a particular
        motor action.
    """

    def __init__(self):
        super().__init__()


class CompositeActionController:
    """
        A composite action's controller contains a slot for every schema. Each slot contains data about whether
        the schema lies along some chain to the goal state, and, if so, the proximity to the goal that will be
        achieved if the schema is activated. (see Drescher, 1991, p. 59)
    """

    def __init__(self):
        """
            To initialize the controller, the mechanism broadcasts a message backwards in parallel through chains of
            schemas that lead to the goal state (see Drescher, 1991, Section 5.1.2).
        """
        pass


class CompositeAction(Action):
    """
        “a composite action is essentially a subroutine, defined by a goal state and implemented by component schemas
        coordinated by a controller” (see Drescher, 1991, p. 90) It specifies a high-level action plan intended to
        bring about some goal state.
    """

    def __init__(self):
        super().__init__()

        # a set of positively or negatively included items that specify some set of environmental conditions
        self.goal_state = None

        # a set of schemas that are part of a chain of schemas used to achieve the goal state of this composite action
        # FIXME: This data structure is probably wrong. It likely needs to be a directed graph.
        self.components = list()

        self.controller = CompositeActionController()

    def is_enabled(self):
        """
            A composite action is "enabled" when one of its components is applicable.
        """
        pass

    # TODO: When a new composite action is defined, a bare schema is constructed for that composite action
    # TODO: (see Drescher, 1991, p. 71)


# TODO: I need a method for creating unique identifiers for actions
def get_unique_id():
    return 1


class ExtendedResult:
    def __init__(self):
        pass


class Schema:
    """
    a three-component data structure used to express a prediction about the environmental state that
    will result from taking a particular action when in a given environmental state (i.e., context).

    Note: A schema is not a rule that says to take a particular action when its context is satisfied;
    the schema just says what might happen if that action were taken.
    """
    INITIAL_RELIABILITY = 0.0
    INITIAL_CORRELATION = 0.0
    INITIAL_DURATION = 0.0
    INITIAL_COST = 0.0

    def __init__(self, context=None, action=None, result=None):
        self._context = context  # this should be a set of items
        self._action = action
        self._result = result  # this should be a set of items

        # A unique identifier for this schema
        self.id = get_unique_id()

        # TODO: May of these properties will depend on other values. It's likely that
        # TODO: the properties will need to be calculated via a @property method call from
        # TODO: these other values.

        # TODO: Should reliability be replaced by base-level activation?
        # A schema's reliability is the likelihood with which the schema succeeds (i.e., its
        # result obtains) when activated
        self.reliability = Schema.INITIAL_RELIABILITY

        # A schema's correlation indicates the extent to which the schema’s result depends on
        # its action (see Drescher, 1991, p.55). It is calculated as the ratio of the frequency
        # of transitioning to a schema’s result with versus without the schema's activation
        self.correlation = Schema.INITIAL_CORRELATION

        # The duration is the average time from the activation to the completion of an action.
        self.duration = Schema.INITIAL_DURATION

        # The cost is the minimum (i.e., the greatest magnitude) of any negative-valued results
        # of schemas that are implicitly activated as a side effect of the given schema’s [explicit]
        # activation on that occasion (see Drescher, 1991, p.55).
        self.cost = Schema.INITIAL_COST

        # TODO: How could these be learned?
        self.overriding_conditions = None

        # a schema’s extended context tried to identify conditions under which the result more
        # reliably follows the action. Each extended context slot keeps track of whether the schema
        # is significantly more reliable when the associated item is On (or Off). When the mechanism
        # thus discovers an item whose state is relevant to the schema’s reliability, it adds that
        # item (or its negation) to the context of a spin-off schema. (see Drescher, 1987, p. 291)
        #
        # Supports the discovery of:
        #
        # 	reliable schemas (see Drescher, 1991, Section 4.1.2)
        # 	overriding conditions (see Drescher, 1991, Section 4.1.5)
        # 	sustained context conditions (see Drescher, 1991, Section 4.1.6)
        # 	conditions for turning Off a synthetic item (see Drescher, 1991, Section 4.2.2)
        self.extended_context = None

        # each extended result slot keeps track of whether the associated item turns On more often
        # if the schema has just been activated than if not. If so, the mechanism attributes that
        # state transition to the action, and builds a spin-off schema, with that item included in
        # the result. (If a schema’s activation makes some item more likely to turn Off, the item’s
        # negation joins the result of a spin-off schema.) (see Drescher, 1987, p. 291)
        #
        # Supports the discovery of:
        # 	reliable schemas (see Drescher, 1991, Section 4.1.2)
        # 	chains of schemas (see Drescher, 1991, Section 5.1.2)
        self.extended_result = None

    def __eq__(self, other):
        if self is other:
            return True
        if isinstance(other, Schema):
            return self.id == other.id
        return NotImplemented

    def __ne__(self, other):
        equal = self.__eq__(other)
        if equal is NotImplemented:
            return NotImplemented
        return not equal

    @property
    def context(self):
        return self._context

    @property
    def action(self):
        return self._action

    @property
    def result(self):
        return self._result

    # TODO: Should this be replaced by a current activation?
    def is_context_satisfied(self):
        """ “A context is satisfied when and only when all of its non-negated items are On,
            and all of its negated items are Off.” (see Drescher, 1991, p. 10)

            TODO: This needs to be relaxed to allow for fuzzy matching on sensory data. It
            could be a more Boolean matching rule for concepts with distinct identifiers,
            though sub-classes of things may make for interesting problems (e.g., context species
            a fruit, and we have an apple which ISA fruit.)
        """
        pass

    # TODO: Should this be in action selection?
    def is_applicable(self):
        """ “A schema is said to be applicable when its context is satisfied and no
             known overriding conditions obtain.” (Drescher, 1991, p.53)
        :return:
        """
        pass

    # TODO: I may not need a separate method for this. This seems like it would be useful during learning
    # TODO: to determine what to update.
    # TODO: There may be an additional complication related to saving the actual context vs schema's context. They
    # TODO: may not be identical (in my implementation), though they would have to be similar.
    def is_valid(self, actual_result):
        """
            An applicable schema is said to be valid at times when its assertion is in fact
            true—that is, at times when the result would indeed obtain if the action were taken.”
            (see Drescher, 1991, p.53)

        :param actual_result:
        :return:
        """
        pass

    # TODO: The implementation for this is unclear. I assume it is based on the schema's reliability and some
    # TODO: threshold. But I need to figure this out.
    def is_reliable(self, threshold):
        """
            “The Schema Mechanism uses only reliable schemas to pursue goals. But the mechanism needs to be sensitive
            to intermittent results, because a reliable effect can seem arbitrarily unreliable until the relevant
            context conditions have been identified.” (see Drescher, 1987, p. 291-292)
        :param threshold: the reliability threshold for determining that a schema is "reliable"
        :return:
        """
        pass

    def is_activated(self, schema):
        """ Returns whether this schema is explicitly or implicitly activated.

        :param schema: the schema that was selected for explicit activation
        :return: True if this schema is (implicitly or explicitly) activated; False otherwise.
        """
        return self.is_explicitly_activated() or self.is_implicitly_activated()

    def is_explicitly_activated(self, schema):
        """ Returns True if this schema was explicitly activated.

            Explicit activation occurs when this schema is selected for activation and its action is initiated
            for execution.

        :param schema: the schema that was selected for explicit activation.
        :return: True if this schema is explicitly activated; False otherwise.
        """
        """ """
        return schema.uid == self.id

    def is_implicitly_activated(self, schema):
        """ Returns True if this schema was implicitly activated.

            Implicit activation occurs as a side effect of another schema's explicit activation, under the following
            conditions (see Drescher, 1991, p.54):

                (1) the schema's context is satisfied
                (2) the schema has the same action as the explicitly activated schema

        :param schema: the schema that was selected for explicit activation
        :return: True if this schema is implicitly activated; False otherwise.
        """
        return self.action == schema.action and self.is_context_satisfied()

    def create_spin_off(self, type, item):
        """ Creates a context or result spin-off schema that includes the supplied item in its context or result.

        :param type: "result" (see Drescher, 1991, p. 71) or "context" (see Drescher, 1991, p. 73)
        :param item: a relevant item to add to the context or result of a spin-off schema
        :return: a spin-off schema based on this one
        """
        pass

        # “A spin-off schema copies the given schema’s context, action, and result,
        # but with the designated item included in the copy’s result (or context)” (see Drescher, 1991, p. 72).


class BareSchema(Schema):
    def __init__(self, action):
        super().__init__(context=None, action=action, result=None)


class SchemaMechanism:
    def __init__(self, primitive_actions):
        self.primitive_actions = primitive_actions

        if len(self.primitive_actions) == 0:
            raise ValueError('At least one primitive action must be specified!')

        # TODO: What is the best data structure here? Think about lookup... (perhaps dictionaries keyed on
        # TODO: context and result? But what about similarity-based comparisons for sensory content?)
        self.schemas = list()

        # for each primitive action, the schema mechanism is initialized with a "bare schema" (i.e.,
        # a schema with an empty context and result. (see Drescher, 1991, p. 71)
        for pa in self.primitive_actions:
            self.add_schema(BareSchema(pa))

    def add_schema(self, schema):
        pass
