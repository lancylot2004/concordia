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

from accelerate import Accelerator
from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib
from mistral_common.protocol.instruct.messages import AssistantMessage
from mistral_common.protocol.instruct.messages import SystemMessage
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from transformers import MistralForCausalLM
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

        logging.basicConfig(level = logging.INFO, format = '[%(levelname)s] %(asctime)s :: %(message)s')

        model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
        gguf_file = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"

        logging.info(f"Loading tokeniser and model {model_id}")
        self._tokeniser = MistralTokenizer.v1()
        self._model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file = gguf_file)

        logging.info("Preparing model using accelerator")
        self._accelerator = Accelerator()
        self._model, self._tokeniser = self._accelerator.prepare(self._model, self._tokeniser)
        logging.info("Model prepared on {self._accelerator.device}")

        # logging.info("Moving model to MPS")
        # self._model.to("mps")

        self._model: MistralForCausalLM

        self._measurements = measurements
        self._channel = channel

    @staticmethod
    def _construct_messages(prompt: str) -> ChatCompletionRequest:
        return ChatCompletionRequest(messages = [
            SystemMessage(content = (
                    'You always continue sentences provided by the user and '
                    'you never repeat what the user already said.'
                )),
            UserMessage(content =
                'Question: Is Jake a turtle?\nAnswer: Jake is ',
            ),
            AssistantMessage(content = 'not a turtle.'),
            UserMessage(content = (
                    'Question: What is Priya doing right now?\n'
                    'Answer: Priya is currently '
            )),
            AssistantMessage(content = 'sleeping.'),
            UserMessage(content = prompt),
        ])

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
        logging.info(f"Sending prompt with {len(prompt)} characters, beginning with: '{prompt[:32]}'.")

        tokens = self._tokeniser.encode_chat_completion(messages).tokens
        logging.info(f"Tokenised prompt with {len(tokens)} tokens.")
        input_ids = torch.tensor([tokens], device = self._accelerator.device)
        outputs = self._model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            seed=seed,
        )
        result = self._tokeniser.decode(outputs[0].tolist())
        logging.info(f"Generated response with {len(result)} characters, beginning with: '{result[:32]}'.")

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

            tokens = self._tokeniser.encode_chat_completion(messages).tokens
            logging.info(f"Testing response {response} with prompt of {len(tokens)} tokens.")

            with torch.no_grad():
                outputs = self._model(input_ids = tokens, labels = tokens)

                loss = outputs.loss
                score = -loss.item()

            logging.info(f"Scored response {response} with score {score}.")
            return score

        logging.info(f"Scoring {len(responses)} responses.")
        scores = np.array([_score_response(r) for r in responses])

        # Find the index of the highest score
        idx = np.argmax(scores)
        max_str = responses[idx]
        logging.info(f"Chose response {max_str} with score {scores[idx]}.")

        return idx, max_str, {r: scores[i] for i, r in enumerate(responses)}

if __name__ == "__main__":
    model = LocalModel()
    print(model.sample_text("Hello, how are you?"))
    print(model.sample_choice("What is the capital of France", ["London", "Paris", "Your Mom"]))
