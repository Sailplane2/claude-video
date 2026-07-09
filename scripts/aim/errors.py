"""Custom exception classes for the aim CLI."""


class AimCliError(Exception):
    """Base exception for all aim CLI errors."""
    pass


class SessionNotFoundError(AimCliError):
    """Session directory does not exist."""

    def __init__(self, session_name: str):
        self.session_name = session_name
        super().__init__(
            f"Session '{session_name}' not found. Use `aim list` to see available sessions."
        )


class SessionExistsError(AimCliError):
    """Session already exists when trying to create."""

    def __init__(self, session_name: str):
        self.session_name = session_name
        super().__init__(
            f"Session '{session_name}' already exists. Use a different name."
        )


class LockActiveError(AimCliError):
    """Session is locked by another writer."""

    def __init__(self, session_name: str, lock_info: dict):
        self.session_name = session_name
        self.lock_info = lock_info
        super().__init__(
            f"Session '{session_name}' is locked by PID {lock_info.get('pid', 'unknown')}, "
            f"acquired {lock_info.get('acquired', 'unknown')}. "
            f"Wait for completion, or run `aim repair {session_name}` to break the lock "
            f"(caution: may corrupt session state)."
        )


class LockStaleError(AimCliError):
    """Session lock is stale and was broken."""
    pass


class ProviderError(AimCliError):
    """Provider API error."""
    pass


class ProviderAuthError(ProviderError):
    """API key missing or invalid."""
    pass


class ProviderRateLimitError(ProviderError):
    """Rate limited by provider."""
    pass


class ProviderOverloadError(ProviderError):
    """Provider server overloaded or unavailable."""
    pass


class ProviderContextLengthError(ProviderError):
    """Context exceeds provider limit."""

    def __init__(self, context_tokens: int, max_tokens: int = None, provider: str = "openai"):
        self.context_tokens = context_tokens
        self.max_tokens = max_tokens
        msg = f"Context {context_tokens} tokens exceeds {provider} limit"
        if max_tokens:
            msg += f" of {max_tokens}"
        msg += ". Run `aim compact <session>` to reduce thread length."
        super().__init__(msg)


class BudgetExceededError(AimCliError):
    """Token budget or cost cap exceeded."""
    pass


class SecretFoundError(AimCliError):
    """Potential secret detected in message content."""

    def __init__(self, matches: list):
        self.matches = matches
        patterns = [m['pattern'] for m in matches[:3]]
        super().__init__(
            f"SECURITY BLOCK: {len(matches)} potential secret(s) detected in message: "
            f"{', '.join(patterns)}. Remove secrets before sending."
        )


class NoAPIKeyError(ProviderAuthError):
    """API key not configured."""

    def __init__(self, provider: str = "openai"):
        env_var = "OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY"
        super().__init__(
            f"{env_var} environment variable not set. Set it and retry, "
            f"or use `aim status` to check configuration."
        )
