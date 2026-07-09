# SmartScholar — Test Protocol (Session 3)

**Project:** SmartScholar — Agentic RAG Research Assistant  
**Document Class:** Internship Deliverable (Session 3)  
**Status:** Initial End-to-End Runs Successfully Completed[cite: 4]

---

## 1. Test Runs by Research Profile

Following the integration of the individual agents, the overall system was tested in its three predefined modes (**Fast**, **Medium**, **Pro**) using a standard test query[cite: 4]. The tests were conducted on a high-performance local development machine (Ollama)[cite: 4].

* **Fast Mode:** Runs successfully[cite: 4]. The computation time is moderate to long, as only abstracts are analyzed[cite: 4].
* **Medium Mode:** Runs successfully[cite: 4]. The computation time increases due to a higher number of search queries and papers[cite: 4].
* **Pro Mode:** Runs successfully[cite: 4]. The computation time is very long, as processing full texts is extremely resource-intensive for local hardware[cite: 4].

**Profiles Conclusion:** The system functions with fundamental stability across all three profiles[cite: 4]. The computation time varies heavily depending on the mode and is on the higher end, even on powerful hardware[cite: 4].

---

## 2. Encountered Errors & Observations

During the initial complete runs, two specific issues were identified that need to be optimized for future versions:

### 2.1 JSON Formatting Issue (Occasional Exceptions)
Program crashes occur irregularly within the agent pipeline (e.g., in the Researcher or Analyst Agent)[cite: 4]. 
* **Cause:** The local LLM occasionally fails to adhere strictly to the requested JSON structure[cite: 4]. 
* **Impact:** The system cannot parse the response and throws an exception, causing the run to abort[cite: 4].

### 2.2 Language Behavior (Quality with Non-English Requests)
The system is fundamentally capable of handling other languages (tested with German)[cite: 4].
* **Observation:** When processing German queries, grammatical and sentence structure errors frequently creep into the final report (Synthesizer)[cite: 4]. Currently, the system performs significantly cleaner and more fluently in English, likely because the underlying prompts and the local model (llama3) are highly optimized for English[cite: 4].