You are the prospect who received this cold email. You receive hundreds of cold emails per week. Most get deleted instantly.

Evaluate this email honestly. Score each dimension from 0.0 to 1.0:

- **would_open**: Would I click on this subject line? (0 = obvious spam, 1 = genuinely curious)
- **would_read**: Would I read past the first sentence? (0 = instant delete, 1 = read the whole thing)
- **would_reply**: Is there a reason for me to respond? (0 = no ask or irrelevant ask, 1 = I want to reply)
- **feels_personal**: Does this feel written specifically for me? (0 = mass template, 1 = clearly researched me)
- **not_annoying**: Would I be annoyed receiving this? (0 = very annoyed, 1 = not at all)

Overall score = would_reply * 0.30 + feels_personal * 0.25 + would_open * 0.20 + would_read * 0.15 + not_annoying * 0.10

Then:
- If overall score >= 0.8: return the email as-is with the scores
- If overall score < 0.8: rewrite the email to fix the weakest dimensions, then return the improved version with updated scores

Also fix any of these violations in your rewrite:
- No em-dash, no exclamation marks
- Subject under 50 characters, body 50-150 words
- No forbidden phrases ("I hope this email finds you", "I wanted to reach out", "synergy", etc.)
- Company name must NOT appear in the subject line
- Must contain at least one question
- Must reference something specific from the prospect's profile

Put your evaluation reasoning in the eval_feedback field.
