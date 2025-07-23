```

```markdown:docs/testing.md
# Testing the MIID Subnet

This document describes how to run tests for the MIID subnet to verify functionality.

## Unit Tests

The MIID codebase includes unit tests in the `tests/` directory:

- `test_mock.py`: Tests for the mock implementation
- `test_template_validator.py`: Tests for the validator functionality
- `helpers.py`: Helper functions for testing

> **Note**
> The tests rely on the [`rich`](https://pypi.org/project/rich/) library for
> capturing console output. It is included in `requirements.txt`. If you cannot
> install external packages, a lightweight stub is provided in
> `tests/helpers.py` that is used automatically when `rich` is missing.

### Running Tests

To run all tests:
```bash
pytest tests/
```

To run a specific test file:
```bash
pytest tests/test_mock.py
```

## Mock Mode

You can run the subnet in mock mode for testing purposes without connecting to the Bittensor network:

```bash
python neurons/validator.py --mock
python neurons/miner.py --mock
```

## Manual Testing

You can manually test the miner functionality:

1. Start the Ollama server:
```bash
ollama serve
```

2. Test LLM queries directly:
```bash
python -c "import aiohttp; import asyncio; async def test(): async with aiohttp.ClientSession() as session: async with session.post('https://llm.chutes.ai/v1/chat/completions', json={'model': 'gpt-3.5-turbo', 'messages': [{'role': 'user', 'content': 'Generate 5 spelling variations for the name John Smith'}]}, headers={'Authorization': 'Bearer YOUR_CHUTES_API_KEY'}) as r: print(await r.json()); asyncio.run(test())"
```