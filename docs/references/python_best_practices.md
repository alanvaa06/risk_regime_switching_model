
# Robust Python: Defining Your Own Types

## Enums
- Use `enum.Enum` and `enum.auto()`. No magic numbers or raw strings for categories.
- Style: `class MyEnum(Enum):` with UPPERCASE member names.
- **Verification**: Compare an enum member to a raw value to confirm type-safety (or strict-check behavior).

## Data classes
- Use `@dataclass`. Prefer `frozen=True` for immutability.
- Type-hint all fields. Use `field(default_factory=...)` for mutable defaults (never mutable default literals).
- **Verification**: Instantiate and print `repr()` to confirm auto-generated representation.

## Classes
- Full encapsulation. Proper `__init__`. Use `__str__`, `__eq__` and other dunders as needed.
- Enforce invariants: the class keeps its data in a valid state.
- **Verification**: Test that violates an invariant fails (class prevents it or raises).

## Protocols and interfaces
- Use `typing.Protocol` for structural subtyping. Avoid deep inheritance.
- Use `@runtime_checkable` only when you need `isinstance` checks.
- A class can satisfy a Protocol without inheriting from it (duck typing with safety).
- **Verification**: Implement a class that matches a Protocol without inheriting it; pass it to a function that expects that Protocol.

## Dependency injection
- Pass dependencies (objects/services) into functions or constructors; do not instantiate them inside the consumer.
- Use type hints to declare the interface of the injected dependency.
- **Verification**: Pass two different implementations of the same dependency to the same consumer and confirm both work.

## Composition
- Prefer "Has-A" over "Is-A". Build complex behavior by combining small, focused objects.
- Avoid god objects; delegate responsibilities to composed parts.
- **Verification**: Swap a composed component and show that behavior changes as intended.

## Static analysis
- Use linters (e.g. pylint, flake8) and formatters (e.g. black). Automate; don’t rely on manual style review.
- Watch cyclomatic complexity and style consistency.
- **Verification**: Run the linter on complex code and refactor until it passes.

## Testing (pytest)
- Use pytest. Write unit tests. Arrange–Act–Assert. Use fixtures for setup.
- Test behavior and properties, not only single data points.
- **Verification**: Run `pytest`; all tests pass and failures have clear messages.

## Property-based testing (hypothesis)
- Use Hypothesis. Define strategies (e.g. `st.integers()`, `st.lists()`) for generated data.
- Aim to find edge cases (e.g. division by zero, empty lists) that examples miss.
- **Verification**: Run a hypothesis test that exposes a bug (e.g. wrong handling of zero or empty input).

## Mutation testing
- Use mutmut or similar. Reason about mutant survival.
- Focus: testing the tests — the suite should detect logic changes.
- **Verification**: Introduce a small bug (mutate code); the test suite must fail.

## Readability and intent (no type hints yet)
- PEP 8 compliant. Meaningful names that imply type (e.g. `user_count`, not `x`).
- Robustness = communicating intent. Code must run without syntax errors.

## Built-in types
- Use explicit assignment to show type (e.g. `x = 5  # int`). Prefer `type()` in prints only when demonstrating.
- Know mutable (list, dict) vs immutable (tuple, str) and how references work.

## Type annotations (PEP 484)
- **Mandatory**: type hints on function arguments and return values.
  - `def greet(name: str) -> str:`
- Annotate variables only when inference is ambiguous. Annotations = machine-verified docs.
- **Verification**: Run `mypy`; fix any type mismatch errors.

## Constraining types
- Use `Optional[T]` (not `Union[T, None]`). Use `Union` and `Literal` where needed. Use `Any` sparingly.
- Explicitly handle `None`: unwrap or check before use. Avoid `Any`; it weakens type checking.
- **Verification**: Every `Optional` is checked/unwrapped before use.

## Collections
- Annotate inner types: `list[int]`, `dict[str, int]` (or `List`, `Dict` from `typing` on older Python). Never raw `list`/`dict` in annotations.
- Prefer abstract types for parameters: `Iterable` or `Sequence` for read-only inputs.
- **Verification**: Iteration matches the annotated element type.

## Type checker config
- Configure `mypy.ini` or `pyproject.toml` (e.g. `--disallow-untyped-defs`, `--no-implicit-optional`).
- **Verification**: Changing a strictness flag should change mypy output as expected.