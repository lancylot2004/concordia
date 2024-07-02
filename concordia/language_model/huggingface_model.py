# @title Language Model - pick your model and provide keys

from collections.abc import Collection, Sequence

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib
from concordia.utils import sampling
from huggingface_hub import login
from typing_extensions import override

_MAX_MULTIPLE_CHOICE_ATTEMPTS = 20
_DEFAULT_TEMPERATURE = 0.5
_DEFAULT_TERMINATORS = ()
_DEFAULT_SYSTEM_MESSAGE = (
    'Continue the user\'s sentences. Never repeat their starts. For example, '
    'when you see \'Bob is\', you should continue the sentence after '
    'the word \'is\'.'
)

class HuggingfaceLanguageModel(language_model.LanguageModel):
  def __init__(
      self,
      model_name: str,
      *,
      system_message: str = _DEFAULT_SYSTEM_MESSAGE,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
  ) -> None:
    self._model_name = model_name
    self._system_message = system_message
    self._terminators = []
    if 'llama' in self._model_name:
      self._terminators.extend([''])

    self._tokenizer = AutoTokenizer.from_pretrained(model_name)
    self._model = AutoModelForCausalLM.from_pretrained(model_name)
    self._model.eval()
    if torch.cuda.is_available():
      self._model.to('cuda')

    self._measurements = measurements
    self._channel = channel

  @override
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = _DEFAULT_TERMINATORS,
      temperature: float = _DEFAULT_TEMPERATURE,
      timeout: float = -1,
      seed: int | None = None,
  ) -> str:
    prompt_with_system_message = f'{self._system_message}\n\n{prompt}'

    terminators = (self._terminators.extend(terminators)
                   if terminators is not None else self._terminators)

    inputs = self._tokenizer(prompt_with_system_message, return_tensors="pt")
    if torch.cuda.is_available():
      inputs = {key: val.to('cuda') for key, val in inputs.items()}

    outputs = self._model.generate(
        inputs['input_ids'],
        max_length=max_tokens,
        # max_length=len(inputs['input_ids'][0]) + max_tokens,
        temperature=temperature,
        do_sample=False,
        eos_token_id=self._tokenizer.eos_token_id
    )

    response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

    if self._measurements is not None:
      self._measurements.publish_datum(
          self._channel,
          {'raw_text_length': len(response)})

    return response

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, dict[str, float]]:
    prompt_with_system_message = f'{self._system_message}\n\n{prompt}'
    sample = ''
    answer = ''
    for attempts in range(_MAX_MULTIPLE_CHOICE_ATTEMPTS):
      # Increase temperature after the first failed attempt.
      temperature = sampling.dynamically_adjust_temperature(
          attempts, _MAX_MULTIPLE_CHOICE_ATTEMPTS)

      sample = self.sample_text(
          prompt_with_system_message,
          temperature=temperature,
          seed=seed,
      )
      answer = sampling.extract_choice_response(sample)
      try:
        idx = responses.index(answer)
      except ValueError:
        continue
      else:
        if self._measurements is not None:
          self._measurements.publish_datum(
              self._channel, {'choices_calls': attempts}
          )
        debug = {}
        return idx, responses[idx], debug

    raise language_model.InvalidResponseError(
        (f'Too many multiple choice attempts.\nLast attempt: {sample}, ' +
         f'extracted: {answer}')
    )
