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

from collections import Collection
from typing import Dict
from typing import FrozenSet
from typing import Iterator
from typing import MutableSet
from typing import Optional
from typing import Tuple
from typing import Union

from anytree import AsciiStyle
from anytree import LevelOrderIter
from anytree import NodeMixin
from anytree import RenderTree

from schema_mechanism.data_structures import Action
from schema_mechanism.data_structures import ItemAssertion
from schema_mechanism.data_structures import ItemPoolStateView
from schema_mechanism.data_structures import NULL_STATE_ASSERT
from schema_mechanism.data_structures import Schema
from schema_mechanism.data_structures import StateAssertion
from schema_mechanism.data_structures import StateElement
from schema_mechanism.util import Observer
from schema_mechanism.util import repr_str


# TODO: This might serve as a more general COMPOSITE (design pattern) that implements part of the Schema interface
class SchemaTreeNode(NodeMixin):
    def __init__(self,
                 context: Optional[StateAssertion] = None,
                 action: Optional[Action] = None,
                 label: str = None) -> None:
        self._context = context
        self._action = action

        self._schemas = set()

        self.label = label

    @property
    def context(self) -> Optional[StateAssertion]:
        return self._context

    @property
    def action(self) -> Optional[Action]:
        return self._action

    @property
    def schemas(self) -> MutableSet[Schema]:
        return self._schemas

    @schemas.setter
    def schemas(self, value) -> None:
        self._schemas = value

    def copy(self) -> SchemaTreeNode:
        """ Returns a new SchemaTreeNode with the same context and action.

        Note: any schemas associated with the original will be lost.

        :return: A SchemaTreeNode with the same context and action as this instance.
        """
        return SchemaTreeNode(context=self.context,
                              action=self.action)

    def __hash__(self) -> int:
        return hash((self._context, self._action))

    def __eq__(self, other) -> bool:
        if isinstance(other, SchemaTreeNode):
            return self._action == other._action and self._context == other._context

    def __str__(self) -> str:
        return self.label if self.label else f'{self._context}/{self._action}/'

    def __repr__(self) -> str:
        return repr_str(self, {'context': self._context,
                               'action': self._action, })


# TODO: How do I prevent the same schema from being added multiple times from spin-offs of different parents

class SchemaTree:
    """ A search tree of SchemaTreeNodes with the following special properties:

    1. Each tree node contains a set of schemas with identical contexts and actions.
    2. Each tree node (except the root) has the same action as their descendants.
    3. Each tree node's depth equals the number of item assertions in its context plus one; for example, the
    tree nodes corresponding to primitive (action only) schemas would have a tree height of one.
    4. Each tree node's context contains all of the item assertions in their ancestors plus one new item assertion
    not found in ANY ancestor. For example, if a node's parent's context contains item assertions 1,2,3, then it
    will contain 1,2,and 3 plus a new item assertion (say 4).

    """

    def __init__(self, primitives: Optional[Collection[Schema]] = None) -> None:
        self._root = SchemaTreeNode(label='root')
        self._nodes: Dict[Tuple[StateAssertion, Action], SchemaTreeNode] = dict()

        self._n_schemas = 0

        if primitives:
            self.add_primitives(primitives)

    @property
    def root(self) -> SchemaTreeNode:
        return self._root

    @property
    def n_schemas(self) -> int:
        return self._n_schemas

    @property
    def height(self) -> int:
        return self.root.height

    def __iter__(self) -> Iterator[SchemaTreeNode]:
        iter_ = LevelOrderIter(node=self.root)
        next(iter_)  # skips root
        return iter_

    def __len__(self) -> int:
        """ Returns the number of SchemaTreeNodes (not the number of schemas).

        :return: The number of SchemaTreeNodes in this tree.
        """
        return len(self._nodes)

    def __contains__(self, s: Union[SchemaTreeNode, Schema]) -> bool:
        if isinstance(s, SchemaTreeNode):
            return (s.context, s.action) in self._nodes
        elif isinstance(s, Schema):
            node = self._nodes.get((s.context, s.action))
            return s in node.schemas if node else False

        return False

    def __str__(self) -> str:
        return RenderTree(self._root, style=AsciiStyle()).by_attr(lambda s: str(s))

    def get(self, schema: Schema) -> SchemaTreeNode:
        """ Retrieves the SchemaTreeNode matching this schema's context and action (if it exists).

        :param schema: the schema (in particular, the context and action) on which this retrieval is based
        :return: a SchemaTreeNode (if found) or raises a KeyError
        """
        return self._nodes[(schema.context, schema.action)]

    def add_primitives(self, primitives: Collection[Schema]) -> None:
        """ Adds primitive schemas to this tree.

        :param primitives: a collection of primitive schemas
        :return: None
        """
        self.add(self.root, frozenset(primitives))

    def add_context_spinoffs(self, source: Schema, spin_offs: Collection[Schema]) -> None:
        """ Adds context spinoff schemas to this tree.

        :param source: the source schema that resulted in these spinoff schemas.
        :param spin_offs: the spinoff schemas.
        :return: None
        """
        self.add(source, frozenset(spin_offs), Schema.SpinOffType.CONTEXT)

    def add_result_spinoffs(self, source: Schema, spin_offs: Collection[Schema]):
        """ Adds result spinoff schemas to this tree.

        :param source: the source schema that resulted in these spinoff schemas.
        :param spin_offs: the spinoff schemas.
        :return: None
        """
        self.add(source, frozenset(spin_offs), Schema.SpinOffType.RESULT)

    # TODO: Change this to use a view rather than state
    # TODO: Rename to something more meaningful
    def find_all_satisfied(self, state: Collection[StateElement], *args, **kwargs) -> Collection[SchemaTreeNode]:
        """ Returns a collection of tree nodes containing schemas with contexts that are satisfied by this state.

        :param state: the state
        :return: a collection of schemas
        """
        matches: MutableSet[SchemaTreeNode] = set()

        nodes_to_process = list(self._root.children)
        while nodes_to_process:
            node = nodes_to_process.pop()
            if node.context.is_satisfied(state, *args, **kwargs):
                matches.add(node)
                if node.children:
                    nodes_to_process += node.children

        return matches

    def is_valid_node(self, node: SchemaTreeNode, raise_on_invalid: bool = False) -> bool:
        if node is self._root:
            return True

        # 1. node is in tree (path from root to node)
        if node not in self:
            if raise_on_invalid:
                raise ValueError('invalid node: no path from node to root')
            return False

        # 2. node has proper depth for context
        if len(node.context) != node.depth - 1:
            if raise_on_invalid:
                raise ValueError('invalid node: depth must equal the number of item assertions in context minus 1')
            return False

        # checks that apply to nodes that contain non-primitive schemas (context + action)
        if node.parent is not self.root:

            # 3. node has same action as parent
            if node.action != node.parent.action:
                if raise_on_invalid:
                    raise ValueError('invalid node: must have same action as parent')
                return False

            # 4. node's context contains all of parents
            if not all({ia in node.context for ia in node.parent.context}):
                if raise_on_invalid:
                    raise ValueError('invalid node: context should contain all of parent\'s item assertions')
                return False

            # 5. node's context contains exactly one item assertion not in parent's context
            if len(node.parent.context) + 1 != len(node.context):
                if raise_on_invalid:
                    raise ValueError('invalid node: context must differ from parent in exactly one assertion.')
                return False

        # consistency checks between node and its schemas
        if node.schemas:

            # 6. actions should be identical across all contained schemas, and equal to node's action
            if not all({node.action == s.action for s in node.schemas}):
                if raise_on_invalid:
                    raise ValueError('invalid node: all schemas must have the same action')
                return False

            # 7. contexts should be identical across all contained schemas, and equal to node's context
            if not all({node.context == s.context for s in node.schemas}):
                if raise_on_invalid:
                    raise ValueError('invalid node: all schemas must have the same context')
                return False

        return True

    def validate(self, raise_on_invalid: bool = False) -> Collection[SchemaTreeNode]:
        """ Validates that all nodes in the tree comply with its invariant properties and returns invalid nodes.

        :return: A set of invalid nodes (if any).
        """
        return set([node for node in LevelOrderIter(self._root,
                                                    filter_=lambda n: not self.is_valid_node(n, raise_on_invalid))])

    def add(self,
            source: Union[Schema, SchemaTreeNode],
            schemas: FrozenSet[Schema],
            spin_off_type: Optional[Schema.SpinOffType] = None) -> SchemaTreeNode:
        """ Adds schemas to this schema tree.

        :param source: the "source" schema that generated the given (primitive or spinoff) schemas, or the previously
        added tree node containing that "source" schema.
        :param schemas: a collection of (primitive or spinoff) schemas
        :param spin_off_type: the schema spinoff type (CONTEXT or RESULT), or None when adding primitive schemas

        Note: The source can also be the tree root. This can be used for adding primitive schemas to the tree. In
        this case, the spinoff_type should be None or CONTEXT.

        :return: the parent node for which the add operation occurred
        """
        if not schemas:
            raise ValueError('Schemas to add cannot be empty or None')

        try:
            node = source if isinstance(source, SchemaTreeNode) else self.get(source)
            if Schema.SpinOffType.RESULT is spin_off_type:
                # needed because schemas to add may already exist in set reducing total new count
                len_before_add = len(node.schemas)
                node.schemas |= schemas
                self._n_schemas += len(node.schemas) - len_before_add
            # for context spin-offs and primitive schemas
            else:
                for s in schemas:
                    key = (s.context, s.action)

                    # node already exists in tree (generated from different source)
                    if key in self._nodes:
                        continue

                    new_node = SchemaTreeNode(s.context, s.action)
                    new_node.schemas.add(s)

                    node.children += (new_node,)

                    self._nodes[key] = new_node

                    self._n_schemas += len(new_node.schemas)
            return node
        except KeyError:
            raise ValueError('Source schema does not have a corresponding tree node.')


class SchemaMemoryStats:
    def __init__(self):
        self.n_updates = 0


class SchemaMemory(Observer):
    def __init__(self, primitives: Optional[Collection[Schema]] = None):
        super().__init__()

        self._schema_tree = SchemaTree(primitives)
        self._schema_tree.validate(raise_on_invalid=True)

        # register listeners for primitives
        if primitives:
            for schema in primitives:
                schema.register(self)

        # previous and current state
        self._s_prev: Optional[Collection[StateElement]] = None
        self._s_curr: Optional[Collection[StateElement]] = None

        self._stats: SchemaMemoryStats = SchemaMemoryStats()

    def __len__(self) -> int:
        return self._schema_tree.n_schemas

    def __contains__(self, schema: Schema) -> bool:
        return schema in self._schema_tree

    def __str__(self):
        return str(self._schema_tree)

    @staticmethod
    def from_tree(tree: SchemaTree) -> SchemaMemory:
        """ A factory method to initialize a SchemaMemory instance from a SchemaTree.

        Note: This method can be used to initialize SchemaMemory with arbitrary built-in schemas.

        :param tree: a SchemaTree pre-loaded with schemas.
        :return: a tree-initialized SchemaMemory instance
        """
        sm = SchemaMemory()

        sm._schema_tree = tree
        sm._schema_tree.validate(raise_on_invalid=True)

        # register listeners for schemas in tree
        for node in sm._schema_tree:
            for schema in node.schemas:
                schema.register(sm)

        return sm

    @property
    def stats(self) -> SchemaMemoryStats:
        return self._stats

    def update_all(self, schema: Schema, applicable: Collection[Schema], state: Collection[StateElement]) -> None:
        """ Updates schemas based on results of previous action.

        :param schema: the explicitly activated schema
        :param applicable: the set of all applicable schemas (including the explicitly activated schema)
        :param state: the current state (which includes the explicitly activated schema's results)

        :return: None
        """
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

        # TODO: Still need to update the result schemas.
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

    # TODO: Should SchemaTreeNodes be exposed or should this be Schemas instead?
    def all_applicable(self, state: Collection[StateElement]) -> Collection[SchemaTreeNode]:
        # TODO: Where do I add the items to the item pool???
        # TODO: Where do I create the item state view?

        return self._schema_tree.find_all_satisfied(state)

    def receive(self, *args, **kwargs) -> None:
        source: Schema = kwargs['source']
        spin_off_type: Schema.SpinOffType = kwargs['spin_off_type']
        relevant_items: Collection[ItemAssertion] = kwargs['relevant_items']

        spin_offs = frozenset([create_spin_off(source, spin_off_type, ia) for ia in relevant_items])

        # register listeners for spin-offs
        for s in spin_offs:
            s.register(self)

        self._schema_tree.add(source, spin_offs, spin_off_type)


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

# TODO: Globally change spinoff to spin_off, to be consistent with this use
def create_spin_off(schema: Schema, spin_off_type: Schema.SpinOffType, item_assert: ItemAssertion) -> Schema:
    """ Creates a context or result spin-off schema that includes the supplied item in its context or result.

    :param schema: the schema from which the new spin-off schema will be based
    :param spin_off_type: a supported Schema.SpinOffType
    :param item_assert: the item assertion to add to the context or result of a spin-off schema

    :return: a spin-off schema based on this one
    """
    if Schema.SpinOffType.CONTEXT == spin_off_type:
        new_context = (
            StateAssertion(item_asserts=(item_assert,))
            if schema.context is NULL_STATE_ASSERT
            else schema.context.replicate_with(item_assert)
        )
        return Schema(action=schema.action,
                      context=new_context,
                      result=schema.result)

    elif Schema.SpinOffType.RESULT == spin_off_type:
        new_result = (
            StateAssertion(item_asserts=(item_assert,))
            if schema.result is NULL_STATE_ASSERT
            else schema.result.replicate_with(item_assert)
        )
        return Schema(action=schema.action,
                      context=schema.context,
                      result=new_result)

    else:
        raise ValueError(f'Unsupported spin-off mode: {spin_off_type}')


def create_context_spin_off(source: Schema, item_assert: ItemAssertion) -> Schema:
    """ Creates a CONTEXT spin-off schema from the given source schema.

    :param source: the source schema
    :param item_assert: the new item assertion to include in the spin-off's context
    :return: a new context spin-off
    """
    return create_spin_off(source, Schema.SpinOffType.CONTEXT, item_assert)


def create_result_spin_off(source: Schema, item_assert: ItemAssertion) -> Schema:
    """ Creates a RESULT spin-off schema from the given source schema.

    :param source: the source schema
    :param item_assert: the new item assertion to include in the spin-off's result
    :return: a new result spin-off
    """
    return create_spin_off(source, Schema.SpinOffType.RESULT, item_assert)
