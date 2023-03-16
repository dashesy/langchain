"""Base interface that all chains should implement."""
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Extra, Field, validator

import langchain
from langchain.callbacks import get_callback_manager
from langchain.callbacks.base import BaseCallbackManager


class Memory(BaseModel, ABC):
    """Base interface for memory in chains."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    @abstractmethod
    def memory_variables(self) -> List[str]:
        """Input keys this memory class will load dynamically."""

    @abstractmethod
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Return key-value pairs given the text input to the chain."""

    @abstractmethod
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save the context of this model run to memory."""

    @abstractmethod
    def clear(self) -> None:
        """Clear memory contents."""


def _get_verbosity() -> bool:
    return langchain.verbose


class Chain(BaseModel, ABC):
    """Base interface that all chains should implement."""

    memory: Optional[Memory] = None
    callback_manager: BaseCallbackManager = Field(
        default_factory=get_callback_manager, exclude=True
    )
    verbose: bool = Field(
        default_factory=_get_verbosity
    )  # Whether to print the response text

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @property
    def _chain_type(self) -> str:
        raise NotImplementedError("Saving not supported for this chain type.")

    @validator("callback_manager", pre=True, always=True)
    def set_callback_manager(
        cls, callback_manager: Optional[BaseCallbackManager]
    ) -> BaseCallbackManager:
        """If callback manager is None, set it.

        This allows users to pass in None as callback manager, which is a nice UX.
        """
        return callback_manager or get_callback_manager()

    @validator("verbose", pre=True, always=True)
    def set_verbose(cls, verbose: Optional[bool]) -> bool:
        """If verbose is None, set it.

        This allows users to pass in None as verbose to access the global setting.
        """
        if verbose is None:
            return _get_verbosity()
        else:
            return verbose

    @property
    @abstractmethod
    def input_keys(self) -> List[str]:
        """Input keys this chain expects."""

    @property
    @abstractmethod
    def output_keys(self) -> List[str]:
        """Output keys this chain expects."""

    def _validate_inputs(self, inputs: Dict[str, str]) -> None:
        """Check that all inputs are present."""
        missing_keys = set(self.input_keys).difference(inputs)
        if missing_keys:
            raise ValueError(f"Missing some input keys: {missing_keys}")

    def _validate_outputs(self, outputs: Dict[str, str]) -> None:
        if set(outputs) != set(self.output_keys):
            raise ValueError(
                f"Did not get output keys that were expected. "
                f"Got: {set(outputs)}. Expected: {set(self.output_keys)}."
            )

    @abstractmethod
    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Run the logic of this chain and return the output."""

    async def _acall(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Run the logic of this chain and return the output."""
        raise NotImplementedError("Async call not supported for this chain type.")

    def __call__(
        self, inputs: Union[Dict[str, Any], Any], return_only_outputs: bool = False
    ) -> Dict[str, Any]:
        """Run the logic of this chain and add to output if desired.

        Args:
            inputs: Dictionary of inputs, or single input if chain expects
                only one param.
            return_only_outputs: boolean for whether to return only outputs in the
                response. If True, only new keys generated by this chain will be
                returned. If False, both input keys and new keys generated by this
                chain will be returned. Defaults to False.

        """
        inputs = self.prep_inputs(inputs)
        self.callback_manager.on_chain_start(
            {"name": self.__class__.__name__},
            inputs,
            verbose=self.verbose,
        )
        try:
            outputs = self._call(inputs)
        except (KeyboardInterrupt, Exception) as e:
            self.callback_manager.on_chain_error(e, verbose=self.verbose)
            raise e
        self.callback_manager.on_chain_end(outputs, verbose=self.verbose)
        return self.prep_outputs(inputs, outputs, return_only_outputs)

    async def acall(
        self, inputs: Union[Dict[str, Any], Any], return_only_outputs: bool = False
    ) -> Dict[str, Any]:
        """Run the logic of this chain and add to output if desired.

        Args:
            inputs: Dictionary of inputs, or single input if chain expects
                only one param.
            return_only_outputs: boolean for whether to return only outputs in the
                response. If True, only new keys generated by this chain will be
                returned. If False, both input keys and new keys generated by this
                chain will be returned. Defaults to False.

        """
        inputs = self.prep_inputs(inputs)
        if self.callback_manager.is_async:
            await self.callback_manager.on_chain_start(
                {"name": self.__class__.__name__},
                inputs,
                verbose=self.verbose,
            )
        else:
            self.callback_manager.on_chain_start(
                {"name": self.__class__.__name__},
                inputs,
                verbose=self.verbose,
            )
        try:
            outputs = await self._acall(inputs)
        except (KeyboardInterrupt, Exception) as e:
            if self.callback_manager.is_async:
                await self.callback_manager.on_chain_error(e, verbose=self.verbose)
            else:
                self.callback_manager.on_chain_error(e, verbose=self.verbose)
            raise e
        if self.callback_manager.is_async:
            await self.callback_manager.on_chain_end(outputs, verbose=self.verbose)
        else:
            self.callback_manager.on_chain_end(outputs, verbose=self.verbose)
        return self.prep_outputs(inputs, outputs, return_only_outputs)

    def prep_outputs(
        self,
        inputs: Dict[str, str],
        outputs: Dict[str, str],
        return_only_outputs: bool = False,
    ) -> Dict[str, str]:
        """Validate and prep outputs."""
        self._validate_outputs(outputs)
        if self.memory is not None:
            self.memory.save_context(inputs, outputs)
        if return_only_outputs:
            return outputs
        else:
            return {**inputs, **outputs}

    def prep_inputs(self, inputs: Union[Dict[str, Any], Any]) -> Dict[str, str]:
        """Validate and prep inputs."""
        if not isinstance(inputs, dict):
            _input_keys = set(self.input_keys)
            if self.memory is not None:
                # If there are multiple input keys, but some get set by memory so that
                # only one is not set, we can still figure out which key it is.
                _input_keys = _input_keys.difference(self.memory.memory_variables)
            if len(_input_keys) != 1:
                raise ValueError(
                    f"A single string input was passed in, but this chain expects "
                    f"multiple inputs ({_input_keys}). When a chain expects "
                    f"multiple inputs, please call it by passing in a dictionary, "
                    "eg `chain({'foo': 1, 'bar': 2})`"
                )
            inputs = {list(_input_keys)[0]: inputs}
        if self.memory is not None:
            external_context = self.memory.load_memory_variables(inputs)
            inputs = dict(inputs, **external_context)
        self._validate_inputs(inputs)
        return inputs

    def apply(self, input_list: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Call the chain on all inputs in the list."""
        return [self(inputs) for inputs in input_list]

    def conversation(self, *args: str, **kwargs: str) -> List[str]:
        """Run the chain as text in, text out or multiple variables, text out."""
        if len(self.output_keys) == 2:
            assert "output" in self.output_keys and "intermediate_steps" in self.output_keys
            keep_short = False
            if "keep_short" in kwargs:
                keep_short = kwargs.pop("keep_short")
            outputs = {}
            if args and not kwargs:
                if len(args) != 1:
                    raise ValueError("`run` supports only one positional argument.")
                # outputs = self("Human: " + args[0])
                outputs = self(args[0])
            # if kwargs and not args:
            #     outputs = self(kwargs)
            intermediate = outputs.get("intermediate_steps") or []
            conversation = []
            for action, action_output in intermediate:
                action: str = action.log.strip()
                if not action.startswith(f"AI:"):
                    action = f"AI: {action}"
                if keep_short:
                    # Hide the internal conversation
                    lines = action.split("\n")
                    new_lines = []
                    for l in lines:
                        for term in [" Let me ask for ", " Let me find if ", " Assistant,"]:
                            idx = l.lower().find(term.lower())
                            if idx >= 0:
                                l = l[:idx]
                                if l.lower().strip() == "ai:":
                                    l = ""
                        if not l:
                            continue
                        new_lines.append(l)
                    action = "\n".join(new_lines)
                conversation.append(action)
                if not keep_short or action_output.lstrip().startswith("Here is the edited image"):
                    conversation.append(f"Assistant: {action_output}")
            conversation.append("AI: " + outputs["output"])
            return conversation

        if len(self.output_keys) != 1:
            raise ValueError(
                f"`run` not supported when there is not exactly "
                f"one output key. Got {self.output_keys}."
            )

        if args and not kwargs:
            if len(args) != 1:
                raise ValueError("`run` supports only one positional argument.")
            return ["AI: " + self(args[0])[self.output_keys[0]]]

        if kwargs and not args:
            return ["AI: " + self(kwargs)[self.output_keys[0]]]

        raise ValueError(
            f"`run` supported with either positional arguments or keyword arguments"
            f" but not both. Got args: {args} and kwargs: {kwargs}."
        )

    def run(self, *args: str, **kwargs: str) -> str:
        """Run the chain as text in, text out or multiple variables, text out."""
        if len(self.output_keys) == 2:
            assert "output" in self.output_keys and "intermediate_steps" in self.output_keys
            outputs = {}
            if args and not kwargs:
                if len(args) != 1:
                    raise ValueError("`run` supports only one positional argument.")
                outputs = self(args[0])
            if kwargs and not args:
                outputs = self(kwargs)
            intermediate = outputs.get("intermediate_steps") or []
            assistant = ""
            for action, action_output in intermediate:
                action: str = action.log.strip()
                if not action.startswith(f"AI:"):
                    action = f"AI: {action}"
                action_output = f"Assistant: {action_output}"
                assistant += "\n" + action + "\n" + action_output
            return assistant + "\n" + "AI: " + outputs["output"]
            
        if len(self.output_keys) != 1:
            raise ValueError(
                f"`run` not supported when there is not exactly "
                f"one output key. Got {self.output_keys}."
            )

        if args and not kwargs:
            if len(args) != 1:
                raise ValueError("`run` supports only one positional argument.")
            return self(args[0])[self.output_keys[0]]

        if kwargs and not args:
            return self(kwargs)[self.output_keys[0]]

        raise ValueError(
            f"`run` supported with either positional arguments or keyword arguments"
            f" but not both. Got args: {args} and kwargs: {kwargs}."
        )

    async def arun(self, *args: str, **kwargs: str) -> str:
        """Run the chain as text in, text out or multiple variables, text out."""
        if len(self.output_keys) != 1:
            raise ValueError(
                f"`run` not supported when there is not exactly "
                f"one output key. Got {self.output_keys}."
            )

        if args and not kwargs:
            if len(args) != 1:
                raise ValueError("`run` supports only one positional argument.")
            return (await self.acall(args[0]))[self.output_keys[0]]

        if kwargs and not args:
            return (await self.acall(kwargs))[self.output_keys[0]]

        raise ValueError(
            f"`run` supported with either positional arguments or keyword arguments"
            f" but not both. Got args: {args} and kwargs: {kwargs}."
        )

    def dict(self, **kwargs: Any) -> Dict:
        """Return dictionary representation of chain."""
        if self.memory is not None:
            raise ValueError("Saving of memory is not yet supported.")
        _dict = super().dict()
        _dict["_type"] = self._chain_type
        return _dict

    def save(self, file_path: Union[Path, str]) -> None:
        """Save the chain.

        Args:
            file_path: Path to file to save the chain to.

        Example:
        .. code-block:: python

            chain.save(file_path="path/chain.yaml")
        """
        # Convert file to Path object.
        if isinstance(file_path, str):
            save_path = Path(file_path)
        else:
            save_path = file_path

        directory_path = save_path.parent
        directory_path.mkdir(parents=True, exist_ok=True)

        # Fetch dictionary to save
        chain_dict = self.dict()

        if save_path.suffix == ".json":
            with open(file_path, "w") as f:
                json.dump(chain_dict, f, indent=4)
        elif save_path.suffix == ".yaml":
            with open(file_path, "w") as f:
                yaml.dump(chain_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"{save_path} must be json or yaml")
