from enum import Enum, unique


@unique
class Example(Enum):
    CIRCULAR = "Circular(5)"
    HARD = "Hard(3)"
    HILL = "Hill(10)"
    LONG_MONOMIAL = "Long Monomial(3)"

    def __str__(self):
        return self.value

    def to_system(self) -> str:
        match self:
            case Example.CIRCULAR:
                return "x' = y^5\n" \
                       "y' = x^5"
            case Example.HILL:
                return "h' = 10*i^2 * t^9\n" \
                       "i' = -10*i^2 * t^9\n" \
                       "t' = 1"
            case Example.HARD:
                return "a' = a^2 * b^2 * c^3\n" \
                       "b' = a^2\n" \
                       "c' = b^2"
            case Example.LONG_MONOMIAL:
                return "x0' = x0^2 * x1^2 * x2^2 + x1^2\n" \
                       "x1' = x0^2 * x1^2 * x2^2 + x2^2\n" \
                       "x2' = x0^2 * x1^2 * x2^2 + x0^2"
