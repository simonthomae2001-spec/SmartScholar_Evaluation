from typing import Any, Callable, List, Type

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.types import Model
from llama_index.llms.ollama import Ollama
from pydantic import ValidationError


def structured_predicted_with_retries(llm: Ollama, output_cls: Type[Model],
    messages: List[ChatMessage], max_retries:int = 4, use_complete_chat:bool=False, logger:Callable=None, **logger_args:Any) ->Model:
    """
    Makes a predict call with a given llm using structured output.
    If the llm fails to generate a valid response the validation exception
    is caught and will be handed to the model with the old failed output in
    order to fix it. If the llm exceeds the maximum number of retries the
    the method returns None.

    Parameters
    ----------
    llm (Ollama) : llm used for generating the output
    output_cls (Type[Model]) : data model that the llm should use for it's output
    messages (List[ChatMessage]) : messages to hand to the llm
    max_retries (int) : maximum number of retries for generating output
    use_complete_chat (bool)=False : if set False only the error message will be handed to the llm if it needs to retry.\
    The error message contains the previous output that failed validation.
    Returns : (Model) containing data from llm | (None) if generation fails
    """
    retries = 0
    llm_kwargs = {"format": output_cls.model_json_schema()}

    while retries <= max_retries:
        response = llm.chat(messages, **llm_kwargs)
        try:
            return output_cls.model_validate_json(response.message.content or "")
        except ValidationError as err:
            if logger:
                logger(**logger_args)
            if not use_complete_chat:
                messages.clear()

            messages.append(
                ChatMessage(
                    role = MessageRole.SYSTEM,
                    content= str(err.json)))
            
            retries +=1
    return None