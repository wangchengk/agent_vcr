"""Example: Record a basic agent session."""

from agentvcr import record, StepType

# Record a simulated agent session
with record("example-flight-search") as tape:
    # Simulate agent thinking
    tape.add_step(
        type=StepType.THINK,
        content="I need to search for flights from SFO to NRT for June 2026",
        tokens=45,
    )

    # Simulate tool call
    tape.add_step(
        type=StepType.TOOL_CALL,
        content="search_flights",
        input={"origin": "SFO", "destination": "NRT", "date": "2026-06-15"},
        output={"flights": [{"airline": "ZIPAIR", "price": 420}, {"airline": "ANA", "price": 680}]},
        duration_ms=2300,
        tokens=187,
        metadata={"tool": "search_flights"},
    )

    # Simulate observing results
    tape.add_step(
        type=StepType.OBSERVE,
        content="Found 2 flights. ZIPAIR is cheapest at $420.",
        tokens=32,
    )

    # Simulate error recovery
    tape.add_step(
        type=StepType.ERROR,
        content="Failed to book — seat unavailable",
        metadata={"error_type": "BookingError"},
    )

    tape.add_step(
        type=StepType.RETRY,
        content="Retrying with alternative date 2026-06-16",
        tokens=28,
    )

    # Simulate success
    tape.add_step(
        type=StepType.TOOL_CALL,
        content="book_flight",
        input={"airline": "ZIPAIR", "date": "2026-06-16"},
        output={"booking_id": "BK-20260616-ZP-001"},
        duration_ms=1800,
        tokens=95,
        metadata={"tool": "book_flight"},
    )

    tape.add_step(
        type=StepType.THINK,
        content="Successfully booked! Booking ID: BK-20260616-ZP-001",
        tokens=22,
    )

# Print summary
import sys; sys.stdout.reconfigure(encoding='utf-8', errors='replace')
print(f"\n📼 Recorded {tape.step_count} steps in {tape.duration_ms / 1000:.1f}s")
print(f"   Tool calls: {len(tape.tool_calls)}, Errors: {len(tape.errors)}")
print(f"   Total tokens: {tape.total_tokens}")

# Replay
from agentvcr.player import Player

Player(tape).play()

# Export to HTML
tape.export("example-flight-search.html", format="html")
print("\n✅ Exported to example-flight-search.html")
