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

"""Language Model that uses Together AI api.

Recommended model name is 'google/gemma-2-9b-it'
"""

from collections.abc import Collection, Sequence

from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, Gemma2ForCausalLM, AutoTokenizer
from typing_extensions import override

_DEFAULT_MAX_TOKENS = 5000


class LocalModel(language_model.LanguageModel):
    """Language Model that runs Gemma-2 locally."""

    def __init__(
        self,
        *,
        measurements: measurements_lib.Measurements | None = None,
        channel: str = language_model.DEFAULT_STATS_CHANNEL,
    ):
        """Initializes the instance.

        Args:
        measurements: The measurements object to log usage statistics to.
        channel: The channel to write the statistics to.
        """

        # model_id = "bartowski/gemma-2-9b-it-GGUF"
        # filename = "gemma-2-9b-it-IQ4_XS.gguf"

        model_id = "google/gemma-2-2b-it"

        self._tokeniser = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(model_id)
        self._model.to("mps")

        self._tokeniser: AutoTokenizer
        self._model: Gemma2ForCausalLM

        self._measurements = measurements
        self._channel = channel

    @staticmethod
    def _construct_messages(prompt: str) -> list[dict[str, str]]:
        return [
            {
                'role': 'user',
                'content': (
                    'You always continue sentences provided '
                    'by the user and you never repeat what '
                    'the user has already said. All responses must end with a '
                    'period. Try not to use lists, but if you must, then '
                    'always delimit list items using either '
                    r"semicolons or single newline characters ('\n'), never "
                    r"delimit list items with double carriage returns ('\n\n')."
                    "Question: Is Jake a turtle?\nAnswer: Jake is "
                ),
            },
            {'role': 'assistant', 'content': 'not a turtle.'},
            {
                'role': 'user',
                'content': (
                    'Question: What is Priya doing right now?\nAnswer: '
                    + 'Priya is currently '
                ),
            },
            {'role': 'assistant', 'content': 'sleeping.'},
            {'role': 'user', 'content': prompt},
        ]

    @override
    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
        terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
        temperature: float = language_model.DEFAULT_TEMPERATURE,
        timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
        seed: int | None = None,
    ) -> str:
        # gemma2 does not support `tokens` + `max_new_tokens` > 8193.
        # gemma2 interprets our `max_tokens`` as their `max_new_tokens`.
        max_tokens = min(max_tokens, _DEFAULT_MAX_TOKENS)

        messages = self._construct_messages(prompt)
        print(f"Sending: {len(messages)}, ", end = "")

        ids = self._tokeniser \
            .apply_chat_template(messages, return_tensors = 'pt') \
            .to(self._model.device)
        print(f"with {len(ids)} tokens, ", end = "")
        outputs = self._model.generate(
            ids,
            max_new_tokens = max_tokens,
            do_sample = True,
            temperature = temperature,
            seed = seed,
        )
        result = self._tokeniser.decode(outputs[0], skip_special_tokens = True)
        print(f"completed.")

        if self._measurements is not None:
            self._measurements.publish_datum(
                self._channel,
                {'raw_text_length': len(result)},
            )
        # Remove the occasional sentence fragment from the end of the result.
        last_stop = result.rfind('.')
        if last_stop >= 0:
            result = result[: last_stop + 1]
        return result

    @override
    def sample_choice(
        self,
        prompt: str,
        responses: Sequence[str],
        *,
        seed: int | None = None,
    ) -> tuple[int, str, dict[str, float]]:

        def _score_response(response: str) -> float:
            augmented_prompt = prompt + response
            messages = self._construct_messages(augmented_prompt)

            input_ids = self._tokeniser \
                .apply_chat_template(messages, return_tensors='pt') \
                .to(self._model.device)

            with torch.no_grad():
                outputs = self._model(input_ids=input_ids, labels=input_ids)

                loss = outputs.loss
                score = -loss.item()

            return score

        # Score each response
        scores = np.array([_score_response(r) for r in responses])

        # Find the index of the highest score
        idx = np.argmax(scores)
        max_str = responses[idx]

        return idx, max_str, {r: scores[i] for i, r in enumerate(responses)}
