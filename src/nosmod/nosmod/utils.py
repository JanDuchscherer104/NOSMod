from pathlib import Path
from typing import Any, Callable, Generic, Literal, Type, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field
from pydantic_yaml import parse_yaml_file_as, to_yaml_file
from rich.console import Console as RichConsole

TargetType = TypeVar("TargetType")


class BaseConfig(BaseModel, Generic[TargetType]):
    target: Callable[["BaseConfig[TargetType]"], TargetType]

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    @classmethod
    def from_yaml(
        cls: Type["BaseConfig[TargetType]"], file: Union[Path, str]
    ) -> "BaseConfig[TargetType]":
        # Load a configuration instance from a YAML file
        return cls.model_validate(parse_yaml_file_as(cls, file))  # type: ignore

    def to_yaml(self, file: Union[Path, str]) -> None:
        # Save the current configuration instance to a YAML file
        to_yaml_file(file, self, indent=4)

    def __str__(self) -> str:
        lines = [self.__class__.__name__ + ":"]
        for key, val in self.model_dump().items():
            if isinstance(val, tuple):
                val = "[" + ", ".join(map(str, val)) + "]"
            lines.append(f"{key}: {val}")
        return "\n    ".join(lines)

    def setup_target(self, **kwargs: Any) -> TargetType:
        return self.target(self, **kwargs)

    def inspect(self) -> str:
        lines = [self.__class__.__name__ + ":"]
        for field_name, field in self.model_fields.items():
            lines.append(
                f'{field_name}: (value={getattr(self, field_name)}, type={field.annotation.__name__}, description="{field.description}")'
            )
        return "\n    ".join(lines)


class _Console(RichConsole):
    _instance: RichConsole = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(_Console, cls).__new__(cls)
            cls._instance.__init__(*args, **kwargs)
        return cls._instance


CONSOLE = _Console(width=120)
