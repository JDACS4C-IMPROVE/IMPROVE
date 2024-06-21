from enum import Enum

class StringEnum(Enum):
    """
    Extension of the Enum class that returns the string representation of the enum member.
    """

    def __str__(self):
        return str(self.value)