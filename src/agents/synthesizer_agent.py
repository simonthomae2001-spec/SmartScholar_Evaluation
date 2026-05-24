"""
synthesizer_agent.py — Final literature review synthesis (Step 16).

The Synthesizer takes all approved analysis claims and composes a
formatted scientific Markdown literature review, citing each paper
with its assigned citation ID.

Current implementation: **stub** that produces a template review.
"""

from __future__ import annotations


class SynthesizerAgent:
    """Composes the final literature review from approved claims."""

    def synthesize_review(self, approved_claims: list[dict]) -> str:
        """
        Produce a formatted Markdown literature review.

        Parameters
        ----------
        approved_claims : list[dict]
            Structured analysis records that passed the Critic's
            verification. Each record has keys: ``citation_id``,
            ``methodology``, ``findings``, ``limitations``,
            ``user_relevance``.

        Returns
        -------
        str
            A Markdown-formatted literature review string.
        """
        if not approved_claims:
            return "No approved claims available for synthesis."

        lines: list[str] = [
            "# Literature Review\n",
            "## Overview\n",
            f"This review synthesises findings from "
            f"**{len(approved_claims)} sources**.\n",
            "---\n",
            "## Detailed Analysis\n",
        ]

        for claim in approved_claims:
            cid = claim.get("citation_id", "[?]")
            lines.append(f"### Source {cid}\n")
            lines.append(f"**Methodology:** {claim.get('methodology', 'N/A')}\n")
            lines.append(f"**Key Findings:** {claim.get('findings', 'N/A')}\n")
            lines.append(f"**Limitations:** {claim.get('limitations', 'N/A')}\n")
            lines.append(f"**Relevance:** {claim.get('user_relevance', 'N/A')}\n")
            lines.append("---\n")

        lines.append("## Conclusion\n")
        lines.append(
            "> **Note:** This is a stub synthesis. Replace the "
            "`SynthesizerAgent` with LLM-powered generation for "
            "production-quality reviews.\n"
        )

        return "\n".join(lines)
