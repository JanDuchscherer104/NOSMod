from datetime import datetime
from pathlib import Path
from typing import Type, TypeVar

from pydantic import BaseModel
from pydantic_yaml import parse_yaml_file_as, to_yaml_file

T = TypeVar("T", bound="YamlBaseModel")


class YamlBaseModel(BaseModel):
    @classmethod
    def from_yaml(cls: Type[T], file: Path | str) -> T:
        return cls.model_validate(parse_yaml_file_as(cls, file))  # type: ignore

    def to_yaml(self, file: Path | str) -> None:
        to_yaml_file(file, self, indent=4)
