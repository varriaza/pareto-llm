from pydantic import BaseModel, field_validator


class Defaults(BaseModel):
    runs_per_test: int = 3
    keep_model_files: bool = False


class BenchmarkEntry(BaseModel):
    type: str
    name: str
    config: dict


class BenchmarkConfig(BaseModel):
    run_label: str
    defaults: Defaults = Defaults()
    models: list[str]
    benchmarks: list[BenchmarkEntry]

    @field_validator("models")
    @classmethod
    def models_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("models list must not be empty")
        return v

    @field_validator("benchmarks")
    @classmethod
    def benchmarks_not_empty(cls, v: list[BenchmarkEntry]) -> list[BenchmarkEntry]:
        if not v:
            raise ValueError("benchmarks list must not be empty")
        return v
