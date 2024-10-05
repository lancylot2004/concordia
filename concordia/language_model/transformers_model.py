from collections.abc import Collection, Sequence
import logging
from textwrap import dedent
from threading import Lock

import torch

from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib
from concordia.utils.sampling import extract_choice_response, dynamically_adjust_temperature
from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma2ForCausalLM
from typing_extensions import override


_DEFAULT_MAX_TOKENS = 5000
_MAX_MULTIPLE_CHOICE_ATTEMPTS = 20

class TransformersModel(language_model.LanguageModel):
    """Language Model that runs llama-cpp-python locally."""

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

        model_id = "google/gemma-2-9b-it"

        logging.info(f"Loading tokeniser and model {model_id}")
        self._tokeniser = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(model_id)

        self._model : Gemma2ForCausalLM

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
        with TransformersModel._lock:
            max_tokens = min(max_tokens, _DEFAULT_MAX_TOKENS)

            messages = self._construct_messages(prompt)
            logging.info(f"Sending prompt with {len(prompt)} characters, beginning with: '{TransformersModel._remove_newlines(prompt[:32])}'.")

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
            logging.info(f"Generated response with {len(result)} characters, beginning with: '{TransformersModel._remove_newlines(result[:32])}'.")

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
        with TransformersModel._lock:
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
                logging.info(f"Sending prompt with {len(prompt)} characters, beginning with: '{TransformersModel._remove_newlines(prompt[:32])}'.")

                tokens = self._tokeniser.encode_chat_completion(messages).tokens
                logging.info(f"Tokenised prompt with {len(tokens)} tokens.")
                input_ids = torch.tensor([tokens], device = self._accelerator.device)
                outputs = self._model.generate(
                    input_ids,
                    max_new_tokens=_DEFAULT_MAX_TOKENS,
                    do_sample=True,
                    temperature=temperature,
                    seed=seed,
                )
                result = self._tokeniser.decode(outputs[0].tolist())

                answer = extract_choice_response(result)
                logging.info(f"From response beginning with {TransformersModel._remove_newlines(result)[:32]}, Extracted choice: '{answer}'.")

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
    model = TransformersModel()
    print(model.sample_text("Hello, how are you?"))
    print(model.sample_choice("What is the biggest city in the UK", ["London", "Paris", "Your Mom"]))
