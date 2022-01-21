# TODO: GlobalStatistics need to be updated from a method that triggers all ExtendedContext and ExtendedResult updates
# TODO: What are these global statistics???


# update schema-level stats (n_action_taken or n_action_not_taken) stats

# Update Procedure for Extended Result

# for each item in the extended context
# if item is ON:
# if item was already ON in previous state, do not update positive-transition statistics

# if item is OFF
# if item was already OFF in previous state, do not update negative-transition statistics


# Update procedure for Extended Context

from __future__ import annotations

from collections import defaultdict
from typing import Collection
from typing import Dict
from typing import FrozenSet
from typing import Iterator
from typing import List
from typing import MutableSet
from typing import Optional
from typing import Tuple
from typing import Union

from anytree import AsciiStyle
from anytree import LevelOrderIter
from anytree import Node
from anytree import RenderTree
from anytree import WalkError
from anytree import Walker

from schema_mechanism.data_structures import Action
from schema_mechanism.data_structures import ItemAssertion
from schema_mechanism.data_structures import ItemPoolStateView
from schema_mechanism.data_structures import NULL_STATE_ASSERT
from schema_mechanism.data_structures import Schema
from schema_mechanism.data_structures import StateAssertion
from schema_mechanism.data_structures import StateElement
from schema_mechanism.util import Observer


class SchemaMemoryStats:
    def __init__(self):
        self.n_updates = 0
        self.n_schemas = 0


# TODO: How do I prevent the same schema from being added multiple times from spin-offs of different parents
class SchemaTree:
    """ A search tree of schemas with the following special properties:

    1. Each tree node (other than the root) corresponds to a schema.
    2. The root node's children are primitive (action only) schemas.
    3. A node's children are its context spinoff schemas.
    4. A node's depth in the tree corresponds to the number of item assertions in its context (plus one for the root
    node); for example, primitive schemas (which have empty contexts) have a tree height of one.

    """

    def __init__(self) -> None:
        self._root = Node('root')

    @property
    def root(self) -> Node:
        return self._root

    def __iter__(self) -> Iterator[Schema]:
        iter_ = LevelOrderIter(node=self.root)
        next(iter_)  # skips root
        return iter_

    def __len__(self) -> int:
        return len(self.root.descendants)

    def __contains__(self, n: Union[Schema, Node]) -> bool:
        try:
            w = Walker()
            w.walk(self.root, n)
        except WalkError as e:
            return False

        return True

    @property
    def height(self) -> int:
        return self.root.height

    @property
    def primitive_schemas(self) -> Tuple[Schema, ...]:
        return self._root.children

    def add_all(self, parent: Union[Schema, Node], children: Collection[Schema]) -> None:
        """ Adds a (context) spinoff schema to the tree as a child of its parent.

        :param parent: parent Schema or Node
        :param children: a collection of spin-off schemas for this parent

        :return: None
        """
        if not children:
            raise ValueError('Collection of schemas to add must be non-empty')

        parent.children += tuple(children)

        # TODO: Need to associate result spin-offs with their parents somewhere (but not part of the tree)...

    # TODO: Change this to use a view rather than state
    def find_applicable(self,
                        state: Collection[StateElement],
                        schema: Optional[Schema] = None,
                        *args, **kwargs) -> Collection[Schema]:
        """ Finds a collection of applicable schema that are descendants of the given schema (or root if None given).

        :param state: the state used to determine schema applicability
        :param schema: an optional schema at which to begin the search (root will be used otherwise)

        :return: a collection of applicable schemas (from the descendants of the provided schema or the tree's root)
        """
        node: Schema = schema or self._root
        applicable_schemas = set()

        nodes_to_process = list(node.children) if node is self._root else [node]
        while nodes_to_process:
            node = nodes_to_process.pop()
            if node.is_applicable(state, *args, **kwargs):
                applicable_schemas.add(node)
                if node.children:
                    nodes_to_process += node.children

        return applicable_schemas

    def is_valid_node(self, node: Union[Schema, Node], raise_on_invalid: bool = False) -> bool:
        if node is self._root:
            return True

        # checks that apply to all schemas
        if node not in self:
            if raise_on_invalid:
                raise ValueError('invalid node: no path from node to root')
            return False

        if not isinstance(node, Schema):
            if raise_on_invalid:
                raise ValueError('invalid node: non-root node should be instance of Schema')
            return False

        if len(node.context) != node.depth - 1:
            if raise_on_invalid:
                raise ValueError('invalid node: a node\'s depth should equal its number of item assertions minus 1')
            return False

        # checks for primitive nodes
        if node.parent is self._root:

            if not node.is_primitive():
                if raise_on_invalid:
                    raise ValueError('invalid node: children of root must be a primitive schemas')
                return False

        # checks for non-primitive nodes
        else:

            if node.is_primitive():
                if raise_on_invalid:
                    raise ValueError('invalid node: children of non-root nodes must be non-primitive schemas')
                return False

            if node.action != node.parent.action:
                if raise_on_invalid:
                    raise ValueError('invalid node: non-primitive schemas should have same the action as their parents')
                return False

            if not all({ia in node.context for ia in node.parent.context}):
                if raise_on_invalid:
                    raise ValueError('invalid node: non-primitive schemas should have all parent\'s item assertions')
                return False

            if len(node.parent.context) + 1 != len(node.context):
                if raise_on_invalid:
                    raise ValueError('invalid node: children may differ from parents in only one context assertion')
                return False

        return True

    def validate(self, raise_on_invalid: bool = False) -> Collection[Node]:
        """ Validates that all nodes in the tree comply with its invariant properties and returns invalid nodes.

        :return: A set of invalid nodes (if any).
        """
        return set([node for node in LevelOrderIter(self._root,
                                                    filter_=lambda n: not self.is_valid_node(n, raise_on_invalid))])

    def __str__(self) -> str:
        return RenderTree(self._root, style=AsciiStyle()).by_attr(lambda s: str(s))


class SchemaMemory(Observer):
    def __init__(self, primitive_schemas: Collection[Schema]):
        super().__init__()

        if not primitive_schemas:
            raise ValueError('SchemaMemory must have at least one primitive schema.')

        self._schema_tree: SchemaTree = SchemaTree()

        # TODO: Is this necessary???
        self._context_to_schemas_dict: Dict[StateAssertion, MutableSet[Schema]] = defaultdict(lambda: set())

        # TODO: Is this necessary???
        self._action_to_schemas_dict: Dict[Action, List[Schema]] = defaultdict(lambda: list())

        # previous and current state
        self._s_prev: Optional[Collection[StateElement]] = None
        self._s_curr: Optional[Collection[StateElement]] = None

        self._stats: SchemaMemoryStats = SchemaMemoryStats()

        # add primitive schemas
        self._schema_tree.add_all(self._schema_tree.root, primitive_schemas)
        self._context_to_schemas_dict[NULL_STATE_ASSERT] |= set(self._schema_tree.primitive_schemas)

        for ps in self._schema_tree.primitive_schemas:
            self._action_to_schemas_dict[ps.action].append(ps)

        self._stats.n_schemas += len(self._schema_tree)

        self._schema_tree.validate(raise_on_invalid=True)

    def __len__(self) -> int:
        return self._stats.n_schemas

    @property
    def stats(self) -> SchemaMemoryStats:
        return self._stats

    def __contains__(self, schema: Schema) -> bool:
        return schema in self._schema_tree

    def update_all(self, schema: Schema, applicable: Collection[Schema], state: Collection[StateElement]) -> None:
        """ Updates schemas based on results of previous action.

        :param schema: the explicitly activated schema
        :param applicable: the set of all applicable schemas (including the explicitly activated schema)
        :param state: the current state (which includes the explicitly activated schema's results)

        :return: None
        """

        # TODO: Update item pool from state???

        # update current and previous state attributes
        self._s_prev = self._s_curr
        self._s_curr = state

        # create previous and current state views
        v_prev = ItemPoolStateView(self._s_prev)
        v_curr = ItemPoolStateView(self._s_curr)

        # create new and lost state element collections
        new = new_state(self._s_prev, self._s_curr)
        lost = lost_state(self._s_prev, self._s_curr)

        # TODO: My guess is that statistics updates should only occur for non-activated schemas that are applicable. This
        # TODO: is based on the assumption that all of the probabilities within a schema are really conditioned on the context
        # TODO: being satisfied, even though this fact is implicit.

        # update global statistics
        self._stats.n_updates += len(applicable)

        for app in applicable:
            # activated
            if app.action is schema.action:
                app.update(activated=True,
                           v_prev=v_prev,
                           v_curr=v_curr,
                           new=new,
                           lost=lost)

            # non-activated
            else:
                app.update(activated=False,
                           v_prev=v_prev,
                           v_curr=v_curr,
                           new=new,
                           lost=lost)

    # TODO: Change to using a state view
    def all_activated(self, action: Action, state: Collection[StateElement]) -> Collection[Schema]:
        """ Returns all implicitly activated schemas.

            Implicit activation occurs when:

                (1) the schema's context is satisfied
                (2) the schema has the same action as the explicitly activated schema

            (See Drescher, 1991, p.54):

        :param action: the explicitly activated schema's action
        :param state: the current state (which was used in activation)

        :return: a collection of activated schemas
        """
        # the primitive schema corresponding to this action
        primitive_schema = self._action_to_schemas_dict[action][0]
        return self._schema_tree.find_applicable(state, primitive_schema)

    # TODO: Change to using a state view
    def all_applicable(self, state: Collection[StateElement]) -> Collection[Schema]:
        return self._schema_tree.find_applicable(state)

    def receive(self, *args, **kwargs) -> None:
        source: Schema = kwargs['source']
        mode: Schema.SpinOffType = kwargs['mode']
        relevant_items: Collection[ItemAssertion] = kwargs['relevant_items']

        # create spin_off schemas
        spinoffs = [create_spin_off(schema=source, mode=mode, item_assert=ia) for ia in relevant_items]

        # update global statistics
        self._stats.n_schemas += len(spinoffs)

        if mode is Schema.SpinOffType.CONTEXT:
            # add spinoff schemas to schema tree (for CONTEXT mode only)
            self._schema_tree.add_all(parent=source, children=spinoffs)

        elif mode is Schema.SpinOffType.RESULT:
            # add spinoff schemas to result dictionary (for RESULT mode only)
            self._context_to_schemas_dict[source.context] |= spinoffs

        # add schemas to action dictionary
        self._action_to_schemas_dict[source.action] |= spinoffs

    def __str__(self):
        return str(self._schema_tree)


class SchemaSelection:
    """
        See Drescher, 1991, section 3.4
    """
    pass


def held_state(s_prev: Collection[StateElement], s_curr: Collection[StateElement]) -> FrozenSet[StateElement]:
    """ Returns the set of state elements that are in both previous and current state

    :param s_prev: a collection of the previous state's elements
    :param s_curr: a collection of the current state's elements

    :return: a set containing state elements shared between current and previous state
    """
    if not all((s_prev, s_curr)):
        return frozenset()

    return frozenset([se for se in s_curr if se in s_prev])


def new_state(s_prev: Optional[Collection[StateElement]],
              s_curr: Optional[Collection[StateElement]]) -> FrozenSet[StateElement]:
    """ Returns the set of state elements that are in current state but not previous

    :param s_prev: a collection of the previous state's elements
    :param s_curr: a collection of the current state's elements

    :return: a set containing new state elements
    """
    if not all((s_prev, s_curr)):
        return frozenset()

    return frozenset([se for se in s_curr if se not in s_prev])


def lost_state(s_prev: Optional[Collection[StateElement]],
               s_curr: Optional[Collection[StateElement]]) -> FrozenSet[StateElement]:
    """ Returns the set of state elements that are in previous state but not current

    :param s_prev: a collection of the previous state's elements
    :param s_curr: a collection of the current state's elements

    :return: a set containing lost state elements
    """
    if not all((s_prev, s_curr)):
        return frozenset()

    return frozenset([se for se in s_prev if se not in s_curr])


# TODO: Need a way to suppress the creation of a new spin-off schema when a new relevant item is detected, but that
# TODO: schema already exists. Seems like schema comparisons will be necessary, but maybe there is a better way. Some
# TODO: kind of a graph traversal may also be possible, where the graph contains the "family tree" of schemas

def create_spin_off(schema: Schema, mode: Schema.SpinOffType, item_assert: ItemAssertion) -> Schema:
    """ Creates a context or result spin-off schema that includes the supplied item in its context or result.

    :param schema: the schema from which the new spin-off schema will be based
    :param mode: a supported Schema.SpinOffType
    :param item_assert: the item assertion to add to the context or result of a spin-off schema

    :return: a spin-off schema based on this one
    """
    if Schema.SpinOffType.CONTEXT == mode:
        new_context = (
            StateAssertion(item_asserts=(item_assert,))
            if schema.context is NULL_STATE_ASSERT
            else schema.context.replicate_with(item_assert)
        )
        return Schema(action=schema.action,
                      context=new_context,
                      result=schema.result,
                      spin_off_type=Schema.SpinOffType.CONTEXT)

    elif Schema.SpinOffType.RESULT == mode:
        new_result = (
            StateAssertion(item_asserts=(item_assert,))
            if schema.result is NULL_STATE_ASSERT
            else schema.result.replicate_with(item_assert)
        )
        return Schema(action=schema.action,
                      context=schema.context,
                      result=new_result,
                      spin_off_type=Schema.SpinOffType.RESULT)

    else:
        raise ValueError(f'Unsupported spin-off mode: {mode}')
