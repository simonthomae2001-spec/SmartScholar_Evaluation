"""
critic_agent.py — Fact verification and quality gate (Steps 14 & 15).

The Critic examines the Analyst's structured output and decides
whether the analysis meets quality standards or needs revision.
If revision is needed AND the loop budget allows, the pipeline
cycles back to the Analyst.

Current implementation: **stub** that approves on even loop counts
and rejects on odd ones, so the loop actually exercises at least once.
"""

from __future__ import annotations


class CriticAgent:
    """Verifies the quality of the Analyst's structured analysis."""

    def verify_facts(
        self,
        analysis_data: list[dict],
        config: dict,
        loop_count: int,
    ) -> tuple[bool, str]:
        """
        Verify the analysis output and decide pass / fail.

        Parameters
        ----------
        analysis_data : list[dict]
            Structured records from the AnalystAgent.
        config : dict
            Profile config dict (from ``get_config()``).
            Must contain ``max_loops``.
        loop_count : int
            How many critic→analyst iterations have already occurred.

        Returns
        -------
        tuple[bool, str]
            ``(True, feedback)``  → analysis **passed** verification.
            ``(False, feedback)`` → analysis **failed**; if
            ``loop_count < config['max_loops']``, the orchestrator
            should route back to the Analyst.
        """
        max_loops = config["max_loops"]

        # Guard: if we've exhausted our loop budget, auto-approve
        if loop_count >= max_loops:
            return (
                True,
                f"Loop budget exhausted ({loop_count}/{max_loops}). "
                f"Auto-approving {len(analysis_data)} claims.",
            )

        # ---- STUB logic: reject on the first pass to exercise the loop ---- #
        if loop_count == 0:
            return (
                False,
                f"First-pass review (loop {loop_count}/{max_loops}): "
                f"requesting deeper analysis on {len(analysis_data)} records.",
            )

        # Subsequent passes: approve
        return (
            True,
            f"Verification passed on loop {loop_count}/{max_loops}. "
            f"All {len(analysis_data)} claims approved.",
        )
