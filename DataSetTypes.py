import enum


class DataSetNorm(enum.Enum):
    NORMALIZED = enum.auto()
    NOT_NORMALIZED = enum.auto()


class DataSetLevel(enum.Enum):
    LEVEL_18 = enum.auto()
    LEVEL_19 = enum.auto()
    LEVEL_29 = enum.auto()