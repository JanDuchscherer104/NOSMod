import traceback
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generic, Optional, Type, TypeVar, Union
from warnings import formatwarning, warn

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_yaml import parse_yaml_file_as, to_yaml_file
from rich.console import Console as RichConsole


class _Console(RichConsole):
    _instance: RichConsole = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(_Console, cls).__new__(cls)
            cls._instance.__init__(*args, **kwargs)
        return cls._instance

    def warn(self, message: str) -> None:
        stack = traceback.extract_stack()
        filename, lineno, _, _ = stack[-2]
        self.print(
            f"[bright_yellow]Warning:[/bright_yellow] [yellow]{formatwarning(message, UserWarning, filename, lineno)}[/yellow]"
        )


CONSOLE = _Console(width=120)


TargetType = TypeVar("TargetType")


class BaseConfig(BaseModel, Generic[TargetType]):
    target: Callable = Field(default_factory=lambda: None)

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    @classmethod
    def from_yaml(
        cls: Type["BaseConfig[TargetType]"], file: Union[Path, str]
    ) -> "BaseConfig[TargetType]":
        return cls.model_validate(parse_yaml_file_as(cls, file))  # type: ignore

    def to_yaml(self, file: Union[Path, str]) -> None:
        to_yaml_file(file, self, indent=4)

    def __str__(self) -> str:
        lines = [self.__class__.__name__ + ":"]
        for key, val in self.model_dump().items():
            if isinstance(val, tuple):
                val = "[" + ", ".join(map(str, val)) + "]"
            lines.append(f"{key}: {val}")
        return "\n    ".join(lines)

    def setup_target(self, **kwargs: Any) -> TargetType:
        if not callable(factory := getattr(self.target, "setup_target", self.target)):
            CONSOLE.print(
                f"Target '[bold yellow]{self.target}[/bold yellow]' of type [bold yellow]{factory.__class__.__name__}[/bold yellow] is not callable."
            )
            raise ValueError(
                f"Target '{self.target}' of type {factory.__class__.__name__} is not callable / does not have a 'setup_target' or '__init__' method."
            )

        return factory(self, **kwargs)  # type: ignore

    def inspect(self) -> str:
        lines = [self.__class__.__name__ + ":"]
        for field_name, field in self.model_fields.items():
            lines.append(
                f'{field_name}: (value={getattr(self, field_name)}, type={field.annotation.__name__}, description="{field.description}")'
            )
        return "\n    ".join(lines)

    # @model_validator(mode="after")
    # def validate_redundant_fields(self) -> "BaseConfig[TargetType]":
    #     def check_fields(cls, field_values):
    #         for field_name, field_value in cls.__dict__.items():
    #             if isinstance(field_value, BaseConfig):
    #                 check_fields(field_value.__class__, field_values)
    #             elif field_name not in field_values:
    #                 field_values[field_name] = (field_value, cls)
    #             else:
    #                 existing_value, existing_cls = field_values[field_name]
    #                 if existing_value != field_value:
    #                     if existing_cls.__mro__.index(cls) < existing_cls.__mro__.index(
    #                         existing_cls
    #                     ):
    #                         warn(
    #                             f"Field '{field_name}' has different values in parent class '{cls.__name__}' and child class '{existing_cls.__name__}': "
    #                             f"{field_value} != {existing_value}. Overriding with parent class value."
    #                         )
    #                         field_values[field_name] = (field_value, cls)
    #                     else:
    #                         raise ValueError(
    #                             f"Field '{field_name}' has different values in sibling classes '{existing_cls.__name__}' and '{cls.__name__}': "
    #                             f"{existing_value} != {field_value}"
    #                         )

    #     field_values = {}
    #     for cls in self.__class__.__mro__:
    #         if issubclass(cls, BaseConfig) and cls is not BaseConfig:
    #             check_fields(cls, field_values)

    #     return self


class Stage(Enum):
    TRAIN = ("fit", "train")
    VALIDATE = ("validate", "val")
    TEST = ("test",)

    def __init__(self, *values):
        self.values = values

    def __str__(self):
        return self.values[0]

    @classmethod
    def from_str(cls, value: Optional[str]) -> Optional["Stage"]:
        for member in cls:
            if value in member.values:
                return member
        return None
