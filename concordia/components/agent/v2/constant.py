# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A simple acting component that aggregates contexts from components."""

from concordia.typing import component_v2
from concordia.typing import entity as entity_lib


class Constant(component_v2.EntityComponent):
  """A simple component that returns a constant.
  """

  def __init__(
      self,
      state: str,
  ):
    """Initializes the agent.

    Args:
      state: the state of the component.

    Raises:
      ValueError: If the component order is not None and contains duplicate
        components.
    """
    self._state = state

  def pre_act(
      self,
      unused_action_spec: entity_lib.ActionSpec,
  ) -> str:
    del unused_action_spec
    return self._state
