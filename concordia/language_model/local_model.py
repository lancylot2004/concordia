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
import logging
from textwrap import dedent
from threading import Lock

from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib
from concordia.utils.sampling import extract_choice_response, dynamically_adjust_temperature
import numpy as np
from llama_cpp import Llama
from typing_extensions import override

_DEFAULT_MAX_TOKENS = 5000
_MAX_MULTIPLE_CHOICE_ATTEMPTS = 20

class LocalModel(language_model.LanguageModel):
    """Language Model that runs Gemma-2 locally."""

    _lock = Lock()

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

        logging.basicConfig(level = logging.INFO, format = "[%(levelname)s] %(asctime)s :: %(message)s")

        logging.info(f"Loading Llama model")
        self._model = Llama.from_pretrained(
            repo_id = "bartowski/gemma-2-9b-it-GGUF",
            filename = "gemma-2-9b-it-IQ4_XS.gguf",
            verbose = False,
            n_ctx = 2048,
        )
        self._model : Llama

        self._measurements = measurements
        self._channel = channel

    @staticmethod
    def _construct_messages(prompt: str) -> list[dict[str, str]]:
        return [
            {
                "role": "user",
                "content": (
                    "You always continue sentences provided by the user and "
                    "you never repeat what the user already said. Question: "
                    "Is Jake a turtle?\nAnswer: Jake is "
                )
            },
            { "role": "assistant", "content": "not a turtle." },
            {

                "role": "user",
                "content": (
                    "Question: What is Priya doing right now?\n"
                    "Answer: Priya is currently "
                )
            },
            { "role": "assistant", "content": "sleeping." },
            { "role": "user", "content": prompt }
        ]

    @staticmethod
    def _remove_newlines(text: str) -> str:
        return text.replace("\n", "\\n")

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
        with LocalModel._lock:
            max_tokens = min(max_tokens, _DEFAULT_MAX_TOKENS)

            messages = self._construct_messages(prompt)
            logging.info(f"Sending prompt with {len(prompt)} characters, beginning with: '{LocalModel._remove_newlines(prompt[:32])}'.")

            result = self._model.create_chat_completion(
                messages,
                temperature = temperature,
                seed = seed,
                max_tokens = max_tokens,
            )["choices"][0]["message"]["content"]

            logging.info(f"Generated response with {len(result)} characters, beginning with: '{LocalModel._remove_newlines(result[:32])}'.")

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
        with LocalModel._lock:
            prompt = dedent(f"""
                {prompt}

                Pick one best response:
                {" ".join(response for response in responses)}
            """)

            logging.info(f"Sampling choice out of {responses}.")

            for attempts in range(_MAX_MULTIPLE_CHOICE_ATTEMPTS):
                # Increase temperature after the first failed attempt.
                temperature = dynamically_adjust_temperature(attempts, _MAX_MULTIPLE_CHOICE_ATTEMPTS)

                messages = self._construct_messages(prompt)
                logging.info(f"Sending prompt with {len(prompt)} characters, beginning with: '{LocalModel._remove_newlines(prompt[:32])}'.")

                result = self._model.create_chat_completion(
                    messages,
                    temperature = temperature,
                    seed = seed,
                    max_tokens = _DEFAULT_MAX_TOKENS,
                )["choices"][0]["message"]["content"]

                answer = extract_choice_response(result)
                logging.info(f"From response beginning with {LocalModel._remove_newlines(result)[:32]}, Extracted choice: '{answer}'.")

                try:
                    idx = responses.index(answer)
                except (TypeError, ValueError):
                    continue
                else:
                    if self._measurements is not None:
                        self._measurements.publish_datum(
                            self._channel, {'choices_calls': attempts}
                        )

                    return idx, responses[idx], {}

            raise language_model.InvalidResponseError(
                (f'Too many multiple choice attempts.\nLast attempt: {result}, ' +
                f'extracted: {answer}')
            )

if __name__ == "__main__":
    model = LocalModel()
    print(model.sample_text("Hello, how are you?"))
    print(model.sample_choice("What is the biggest city in the UK", ["London", "Paris", "Your Mom"]))
