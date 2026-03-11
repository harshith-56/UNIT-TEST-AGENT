# AI Test Agent

Reusable GitHub Action and Python implementation for stateless AI-assisted unit test generation in pull request CI jobs.

## Capabilities

- Detects pull request changes from `git diff origin/<base_branch>...HEAD`
- Supports Python, JavaScript, JSX, TypeScript, and TSX source files
- Generates tests for `pytest` and `jest`
- Writes only to `tests/ai_generated/`
- Avoids overwriting existing tests and validates generated syntax before writing
- Runs the repository test suite after test generation

## Required Environment

The action expects GitHub pull request context variables and these LLM settings:

- `LLM_API_URL`
- `LLM_API_KEY`
- `LLM_MODEL`

## Usage

Reference this repository as a GitHub Action from another repository workflow and provide the LLM inputs. The action installs its own Python dependencies and executes the agent inside the workflow job.

## Safety Rules

- Never modifies application source files
- Never deletes tests
- Never writes outside `tests/ai_generated/`
- Never executes LLM-produced shell commands
- Uses only pull request diff context and repository files available in the CI workspace
