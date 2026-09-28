"""Execution evidence: run the tests that exercise the change and compare with
the session's start. (Week 2 of the MVP; for now it only reports what it
would need.)"""
from notyet.findings import Context, EngineResult


def run(ctx: Context) -> EngineResult:
    result = EngineResult()
    if not ctx.config.test_command:
        result.not_checked.append("tests: no test command configured (run `notyet init`)")
        return result
    result.not_checked.append("tests: execution checks aren't implemented yet (MVP week 2)")
    return result
