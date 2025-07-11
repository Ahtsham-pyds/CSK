Guardrails are just methods to prevent out LLM to give undesired output.
This can include-
1. Mitigating hellucination.
2. Removing sensite PII information (leke personal data).
3. Keep the bot on the topic.
4. Remoce Toxic content

How guardrails are implemented. Mostly in 3 ways.
1. Rules Based
   a. Regular Expression.
   b. Pattern Matching
   c. Keyword Filters

2. Small Finetuned ML Models
   a. Classification
   b. Topic Detection
   c. Named Entity Recognition

3. Secondary LLm calls
   a. Score for Toxicity
   b. Rate Tone Of Voice
   c. Verify Coherence




Some guradrails are implemented on the prompt iteself. Like the image shows

<img width="508" height="339" alt="image" src="https://github.com/user-attachments/assets/3a795d9c-b444-47f3-ae2a-9f66aec5cb32" />





Some guardrail are implemnted on the LLm output
<img width="680" height="405" alt="image" src="https://github.com/user-attachments/assets/b9d1d2a2-a874-49f6-a155-aa26c4fc5dfc" />

