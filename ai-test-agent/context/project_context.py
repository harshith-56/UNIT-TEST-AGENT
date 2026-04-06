from __future__ import annotations

from dataclasses import dataclass, field

from context.event_context import EventContext


@dataclass(frozen=True)
class StructuredContext:
    rules: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)

    def combined_rules(self) -> list[str]:
        base_rules = [*self.constraints, *self.rules]

        file_filter_rule = (
                "Only generate tests for functions that contain meaningful logic. "
                "Skip functions that are trivial, such as those with only constants, "
                "simple returns, configuration, routing, or no branching. "
                "Focus only on functions with business logic, validation, "
                "transformations, or multiple execution paths."
                "SKIP these — do NOT generate tests for them:\n"
                "- Data classes or model classes that only set attributes "
                "in __init__ and return dicts in to_dict() with no "
                "branching, no validation, no conditionals\n"
                "- __repr__, __str__, __eq__, __hash__ methods\n"
                "- __init__ methods that only assign self.x = x with "
                "no validation, no conditionals, no external calls\n"
                "- to_dict(), from_dict(), serialize(), deserialize() "
                "methods that only construct and return a dict\n"
                "- Enum classes and their members\n"
                "- Constants, configuration dictionaries, type aliases\n"
                "- Functions whose entire body is a single return statement "
                "with no branching\n"
                "- Property getters that only return self._x\n"
                "ONLY generate tests for functions that have at least ONE "
                "of these: if/else branch, try/except block, loop with "
                "conditional, explicit raise/throw, call to external "
                "service/database/API, or complex computation."
        )

        return [file_filter_rule, *base_rules]


def parse_project_context(raw_context: str) -> StructuredContext:
    if not raw_context or not raw_context.strip():
        return StructuredContext(rules=[], constraints=[])
    return StructuredContext(rules=[raw_context.strip()], constraints=[])


def extract_pr_context(event_context: EventContext) -> StructuredContext:
    parts = [
        p.strip()
        for p in [event_context.pull_request_title, event_context.pull_request_body]
        if p and p.strip()
    ]
    if not parts:
        return StructuredContext()
    return StructuredContext(rules=["\n".join(parts)], constraints=[])
