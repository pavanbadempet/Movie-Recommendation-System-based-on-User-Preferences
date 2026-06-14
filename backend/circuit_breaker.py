"""
Circuit Breaker Pattern for External API Resilience

This module provides:
- Circuit breaker implementation for external API calls
- Automatic failure detection and circuit opening
- Half-open state for testing recovery
- Fallback mechanisms
- Metrics and monitoring integration
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, calls are rejected
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Number of failures before opening
    success_threshold: int = 2  # Number of successes to close circuit
    timeout: float = 60.0  # Seconds before attempting recovery
    expected_exception: Exception = Exception  # Exception type to catch
    fallback: Callable | None = None  # Fallback function
    window_size: int = 100  # Size of sliding window for metrics
    call_timeout: float | None = None  # Timeout for individual calls


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: datetime | None = None
    last_success_time: datetime | None = None
    failure_rate: float = 0.0
    recent_failures: list[datetime] = field(default_factory=list)
    recent_successes: list[datetime] = field(default_factory=list)


class CircuitBreaker:
    """
    Circuit breaker for external API resilience.

    Prevents cascading failures by:
    - Detecting failures and opening the circuit
    - Rejecting calls when circuit is open
    - Testing recovery with half-open state
    - Providing fallback mechanisms
    """

    def __init__(self, name: str, config: CircuitBreakerConfig, prometheus_client: Any | None = None):
        """
        Initialize circuit breaker.

        Args:
            name: Name of the circuit breaker
            config: Circuit breaker configuration
            prometheus_client: Optional Prometheus client for metrics
        """
        self.name = name
        self.config = config
        self.prometheus_client = prometheus_client

        self.state = CircuitState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self._lock = threading.Lock()
        self._half_open_success_count = 0

        logger.info(f"Initialized circuit breaker: {name}")

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result or fallback value

        Raises:
            Exception: If circuit is open and no fallback is provided
        """
        with self._lock:
            if self._should_allow_request():
                self.metrics.total_calls += 1
            else:
                self.metrics.rejected_calls += 1
                self._update_prometheus_metrics()
                if self.config.fallback:
                    return self.config.fallback(*args, **kwargs)
                raise CircuitOpenError(f"Circuit {self.name} is OPEN")

        try:
            # Execute the function with timeout if configured
            if self.config.call_timeout:
                result = self._execute_with_timeout(func, args, kwargs)
            else:
                result = func(*args, **kwargs)

            self._on_success()
            return result

        except self.config.expected_exception as e:
            self._on_failure()
            if self.config.fallback:
                logger.warning(f"Call failed, using fallback: {e}")
                return self.config.fallback(*args, **kwargs)
            raise
        except Exception as e:
            self._on_failure()
            if self.config.fallback:
                logger.warning(f"Unexpected error, using fallback: {e}")
                return self.config.fallback(*args, **kwargs)
            raise

    def _execute_with_timeout(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Execute function with timeout."""
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Call timed out after {self.config.call_timeout}s")

        # Set signal handler
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(self.config.call_timeout))

        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # Cancel alarm
            return result
        finally:
            signal.signal(signal.SIGALRM, old_handler)

    def _should_allow_request(self) -> bool:
        """Check if request should be allowed based on circuit state."""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            # Check if timeout has elapsed
            if self.metrics.last_failure_time:
                elapsed = (datetime.now() - self.metrics.last_failure_time).total_seconds()
                if elapsed >= self.config.timeout:
                    # Transition to half-open
                    self.state = CircuitState.HALF_OPEN
                    self._half_open_success_count = 0
                    logger.info(f"Circuit {self.name} transitioned to HALF_OPEN")
                    return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return True

        return False

    def _on_success(self) -> None:
        """Handle successful call."""
        with self._lock:
            self.metrics.successful_calls += 1
            self.metrics.last_success_time = datetime.now()

            # Update recent successes
            self.metrics.recent_successes.append(datetime.now())
            if len(self.metrics.recent_successes) > self.config.window_size:
                self.metrics.recent_successes.pop(0)

            # Update failure rate
            self._update_failure_rate()

            if self.state == CircuitState.HALF_OPEN:
                self._half_open_success_count += 1
                if self._half_open_success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self._half_open_success_count = 0
                    logger.info(f"Circuit {self.name} transitioned to CLOSED")

            self._update_prometheus_metrics()

    def _on_failure(self) -> None:
        """Handle failed call."""
        with self._lock:
            self.metrics.failed_calls += 1
            self.metrics.last_failure_time = datetime.now()

            # Update recent failures
            self.metrics.recent_failures.append(datetime.now())
            if len(self.metrics.recent_failures) > self.config.window_size:
                self.metrics.recent_failures.pop(0)

            # Update failure rate
            self._update_failure_rate()

            # Check if threshold exceeded
            if len(self.metrics.recent_failures) >= self.config.failure_threshold:
                if self.state != CircuitState.OPEN:
                    self.state = CircuitState.OPEN
                    logger.warning(f"Circuit {self.name} transitioned to OPEN")

            self._update_prometheus_metrics()

    def _update_failure_rate(self) -> None:
        """Update failure rate metric."""
        total = self.metrics.successful_calls + self.metrics.failed_calls
        if total > 0:
            self.metrics.failure_rate = self.metrics.failed_calls / total

    def _update_prometheus_metrics(self) -> None:
        """Update Prometheus metrics if client is available."""
        if self.prometheus_client:
            try:
                # Create metrics if they don't exist
                if not hasattr(self, "_prometheus_gauge_state"):
                    self._prometheus_gauge_state = self.prometheus_client.Gauge(
                        f"circuit_breaker_{self.name}_state", f"Circuit breaker state for {self.name}"
                    )
                    self._prometheus_gauge_failure_rate = self.prometheus_client.Gauge(
                        f"circuit_breaker_{self.name}_failure_rate", f"Failure rate for {self.name}"
                    )
                    self._prometheus_counter_total = self.prometheus_client.Counter(
                        f"circuit_breaker_{self.name}_total_calls", f"Total calls for {self.name}"
                    )
                    self._prometheus_counter_success = self.prometheus_client.Counter(
                        f"circuit_breaker_{self.name}_successful_calls", f"Successful calls for {self.name}"
                    )
                    self._prometheus_counter_failed = self.prometheus_client.Counter(
                        f"circuit_breaker_{self.name}_failed_calls", f"Failed calls for {self.name}"
                    )
                    self._prometheus_counter_rejected = self.prometheus_client.Counter(
                        f"circuit_breaker_{self.name}_rejected_calls", f"Rejected calls for {self.name}"
                    )

                # Update metrics
                state_value = {CircuitState.CLOSED: 0, CircuitState.OPEN: 1, CircuitState.HALF_OPEN: 2}[self.state]

                self._prometheus_gauge_state.set(state_value)
                self._prometheus_gauge_failure_rate.set(self.metrics.failure_rate)
                self._prometheus_counter_total.inc(self.metrics.total_calls)
                self._prometheus_counter_success.inc(self.metrics.successful_calls)
                self._prometheus_counter_failed.inc(self.metrics.failed_calls)
                self._prometheus_counter_rejected.inc(self.metrics.rejected_calls)
            except Exception as e:
                logger.error(f"Failed to update Prometheus metrics: {e}")

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.metrics = CircuitBreakerMetrics()
            self._half_open_success_count = 0
            logger.info(f"Circuit breaker {self.name} reset to CLOSED")

    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self.state

    def get_metrics(self) -> dict[str, Any]:
        """Get circuit breaker metrics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "total_calls": self.metrics.total_calls,
            "successful_calls": self.metrics.successful_calls,
            "failed_calls": self.metrics.failed_calls,
            "rejected_calls": self.metrics.rejected_calls,
            "failure_rate": self.metrics.failure_rate,
            "last_failure_time": self.metrics.last_failure_time.isoformat() if self.metrics.last_failure_time else None,
            "last_success_time": self.metrics.last_success_time.isoformat() if self.metrics.last_success_time else None,
        }


class CircuitOpenError(Exception):
    """Exception raised when circuit is open."""

    pass


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    success_threshold: int = 2,
    timeout: float = 60.0,
    expected_exception: Exception = Exception,
    fallback: Callable | None = None,
    call_timeout: float | None = None,
):
    """
    Decorator for circuit breaker protection.

    Args:
        name: Name of the circuit breaker
        failure_threshold: Number of failures before opening
        success_threshold: Number of successes to close circuit
        timeout: Seconds before attempting recovery
        expected_exception: Exception type to catch
        fallback: Fallback function
        call_timeout: Timeout for individual calls
    """

    def decorator(func):
        # Create circuit breaker instance
        config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout=timeout,
            expected_exception=expected_exception,
            fallback=fallback,
            call_timeout=call_timeout,
        )
        breaker = CircuitBreaker(name, config)

        @wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)

        # Attach circuit breaker to wrapper for access
        wrapper.circuit_breaker = breaker
        return wrapper

    return decorator


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def register(self, breaker: CircuitBreaker) -> None:
        """Register a circuit breaker."""
        with self._lock:
            self._breakers[breaker.name] = breaker
            logger.info(f"Registered circuit breaker: {breaker.name}")

    def get(self, name: str) -> CircuitBreaker | None:
        """Get a circuit breaker by name."""
        return self._breakers.get(name)

    def get_all_metrics(self) -> dict[str, dict[str, Any]]:
        """Get metrics for all circuit breakers."""
        return {name: breaker.get_metrics() for name, breaker in self._breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
        logger.info("Reset all circuit breakers")


# Global circuit breaker registry
registry = CircuitBreakerRegistry()


def create_circuit_breaker(
    name: str, config: CircuitBreakerConfig | None = None, prometheus_client: Any | None = None
) -> CircuitBreaker:
    """
    Create and register a circuit breaker.

    Args:
        name: Name of the circuit breaker
        config: Circuit breaker configuration
        prometheus_client: Optional Prometheus client

    Returns:
        Circuit breaker instance
    """
    if config is None:
        config = CircuitBreakerConfig()

    breaker = CircuitBreaker(name, config, prometheus_client)
    registry.register(breaker)
    return breaker


# Pre-configured circuit breakers for common external services
def create_tmdb_circuit_breaker(prometheus_client: Any | None = None) -> CircuitBreaker:
    """Create circuit breaker for TMDB API."""
    config = CircuitBreakerConfig(
        failure_threshold=3, success_threshold=2, timeout=30.0, expected_exception=Exception, call_timeout=10.0
    )
    return create_circuit_breaker("tmdb_api", config, prometheus_client)


def create_openrouter_circuit_breaker(prometheus_client: Any | None = None) -> CircuitBreaker:
    """Create circuit breaker for OpenRouter API."""
    config = CircuitBreakerConfig(
        failure_threshold=5, success_threshold=2, timeout=60.0, expected_exception=Exception, call_timeout=30.0
    )
    return create_circuit_breaker("openrouter_api", config, prometheus_client)


def create_redis_circuit_breaker(prometheus_client: Any | None = None) -> CircuitBreaker:
    """Create circuit breaker for Redis."""
    config = CircuitBreakerConfig(
        failure_threshold=5, success_threshold=3, timeout=10.0, expected_exception=Exception, call_timeout=5.0
    )
    return create_circuit_breaker("redis", config, prometheus_client)
