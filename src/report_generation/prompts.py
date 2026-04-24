"""Prompt templates for interaction and agent analysis."""


def get_interaction_analysis_prompt(row: dict) -> str:
    """
    Generate the prompt for interaction analysis.

    Args:
        row: Dictionary containing interaction data

    Returns:
        Formatted prompt string
    """
    context = f"""
    This is information from CCTS complaints
    - Account Number: {row['attr_account_number']}
    - Conversation Date: {row['calendar_date']}
    - Case Number: {row['Case Number']}
    - Brand: {row['Brand']}
    - Product Line: {row['Product Line']} 
    - Product: {row['Product']}
    - Media Type: {row['media_type']}
    - Customer Issue: {row['Customer Issue']}
    - Potential Root Cause: {row['Root Cause']}
    - Notes: {row['Notes']}
    """

    prompt = f"""
    You are an objective QA and complaint-analysis assistant.

    You will receive:
    1. Background information about a CCTS complaint
    2. A customer service interaction transcript between a customer and an agent

    Your task is to analyze the interaction and determine how it relates to the CCTS complaint.

    CCTS Complaint:
    {context}

    Transcript:
    {row['full_transcript']}

    INSTRUCTIONS:
    - Evaluate the interaction ONLY using:
    1. the transcript
    2. the complaint context provided above
    - Do NOT infer intent, policy, root cause, or facts unless they are explicitly supported by the transcript or complaint context.
    - Do NOT speculate.
    - If the evidence is insufficient, say so clearly.
    - Be balanced: include both helpful agent actions and any gaps or unresolved issues.
    - Focus on observable statements, actions, process steps, outcomes, and customer experience signals.
    - Support important conclusions with direct transcript quotes whenever possible.
    - If a field cannot be determined from the transcript or complaint context, use:
    - "NA" for string fields
    - [] for list fields

    ANALYSIS APPROACH:
    1. Identify the interaction context and what triggered the contact.
    2. Summarize the interaction objectively.
    3. Determine the issue most closely related to the CCTS complaint.
    4. Identify the customer's goal or intent.
    5. Evaluate how the agent responded, including any investigation, explanation, troubleshooting, adjustments, offers, commitments, or limitations.
    6. Assess escalation risk based only on observable evidence.
    7. Identify customer frustration points, unresolved items, and any broader customer journey value.
    8. If complaint correlation is weak, partial, or unclear, reflect that honestly instead of forcing a match.

    OUTPUT REQUIREMENTS:
    - Return ONLY a valid JSON object.
    - Do NOT include markdown fences.
    - Do NOT include comments.
    - Do NOT include any text before or after the JSON.
    - Ensure the JSON is syntactically valid.
    - Escape quotes properly inside strings.
    - Use the schema exactly as provided below.
    - Use only the allowed enum values where specified.
    - If transfer, dropped call, or callback is not explicitly shown, use "No".

    VALID ENUM VALUES:
    - escalation_risk_score: "low", "medium", "high"
    - journey_stage: "initial_contact", "investigation", "resolution_attempt", "escalation", "final_resolution"
    - internal_transfer: "Yes", "No"
    - dropped_calls: "Yes", "No"
    - customer_callbacks: "Yes", "No"

    FIELD GUIDANCE (with N/A Rules):
    - key_topics_discussed: Include only concrete topics actually discussed that relate to the complaint. Use ["N/A"] if no topics relate to the complaint.
    - notable_moments: Use short direct quotes where available; otherwise use concise evidence-based paraphrases. Use ["N/A"] if no notable moments are evident.
    - structured_summary: Include issue raised, customer concern, relevant process or policy references if explicitly discussed, actions taken, offers/credits/discounts/commitments, and interaction outcome. Use "N/A" if interaction content is insufficient.
    - identified_main_issue: State the specific complaint-related issue reflected in the interaction as precisely as possible. Use "N/A" if issue cannot be clearly identified from transcript.
    - customer_intent: Describe what the customer was trying to achieve in this contact. Use "N/A" if customer's intent is unclear or not expressed.
    - agent_response: Describe what the agent actually did or communicated. Use "N/A" if agent response cannot be documented from transcript.
    - contributing_factors: Include only factors that reasonably increased escalation risk and are supported by evidence. Use [] if no contributing factors are evident.
    - customer_frustration_points: Include only explicit or clearly observable frustration signals. Use ["N/A"] if customer showed no frustration.
    - patterns_identified: Include recurring themes only if supported by the transcript or complaint context. Use ["N/A"] if no patterns are evident.
    - interaction_value_for_journey_analysis: Explain why this interaction matters in the broader complaint journey. Use "N/A" if interaction has minimal journey value.
    - unresolved_items: List items that appear to remain open at the end of the interaction. Use ["N/A"] if all items are resolved.
    - recommended_journey_tags: Use short standardized tags based on the interaction evidence only. 

    Return the following JSON structure exactly:

    {{
    "interaction_metadata": {{
        "interaction_date": "{row['calendar_date']}",
        "case_number": "{row['Case Number']}",
        "file_number": "{row['File Number']}",
        "ccts_customer_issue": "{row['Customer Issue']}",
        "key_topics_discussed": [
        "Topic directly related to the CCTS complaint"
        ],
        "notable_moments": [
        "Direct quote or short evidence-based paraphrase of a key transcript moment related to the complaint"
        ]
    }},
    "interaction_summary": {{
        "structured_summary": "Concise but detailed summary of the interaction, including issue raised, customer concern, any relevant process or policy explicitly discussed, actions taken, offers, credits, discounts, commitments, and outcome.",
        "identified_main_issue": "Specific issue that most closely matches the CCTS complaint, including details such as billing error type, disputed amount, service issue, contract concern, cancellation issue, delay, or communication gap.",
        "customer_intent": "What the customer was trying to achieve in this interaction.",
        "agent_response": "How the agent handled the interaction, including explanations, troubleshooting, investigation, offers, adjustments, or limitations communicated."
    }},
    "escalation_factors": {{
        "escalation_risk_score": "low",
        "contributing_factors": [
        "Factor that increased escalation risk"  
        ],
        "customer_frustration_points": [
        "Specific moment or issue that appeared to frustrate the customer"
        ],
        "internal_transfer": "Yes",
        "dropped_calls": "No",
        "customer_callbacks": "No"
    }},
    "journey_insights": {{
        "journey_stage": "initial_contact",
        "patterns_identified": [
        "Recurring issue, behavior, or process pattern observed"
        ],
        "interaction_value_for_journey_analysis": "Why this interaction matters in the broader customer journey, for example first clear statement of issue, failed resolution attempt, repeated explanation, expectation-setting moment, ownership gap, or complaint precursor.",
        "unresolved_items": [
        "Issue that appears to remain open at the end of the interaction"
        ],
        "recommended_journey_tags": [
        "billing_dispute",
        "repeat_contact",
        "explanation_only"
        ]
    }}
    }}
    """
    return prompt


def get_agent_evaluation_prompt(row: dict, context: str) -> str:
    """
    Generate the prompt for agent evaluation.

    Args:
        row: Dictionary containing interaction data
        context: Context string from create_context_prompt

    Returns:
        Formatted prompt string
    """
    agent_eval = f"""You are an objective AI quality assurance analyst specializing in call center agent performance evaluation.

    Your task is to evaluate agent performance using ONLY the transcript and complaint context provided below.

    TRANSCRIPT TO ANALYZE:
    {row['full_transcript']}

    CCTS COMPLAINT:
    {context}

    PRIMARY OBJECTIVE:
    Provide a fair, evidence-based assessment of the agent's performance. Base all conclusions strictly on observable statements and actions in the transcript. Do not infer intent, internal policy exceptions, customer history, or missing context unless explicitly stated.

    GENERAL RULES:
    - Use only transcript evidence and the complaint context provided.
    - Do not guess, assume, or speculate.
    - If evidence is insufficient for a rating or conclusion, explicitly say so.
    - Do not treat minor imperfections as infractions unless supported by meaningful evidence.
    - Be neutral, factual, and concise.
    - Focus on observable behavior, not personality or motive.
    - If the transcript contains multiple agents, evaluate only the agent associated with employee ID "{row['emp_id']}" when possible.
    - If multiple agents appear and attribution is unclear, state that clearly and limit conclusions to what can be confidently attributed.
    - If no evidence supports a concern, mark it as "No" rather than inventing a problem.
    - If it is impossible to determine whether something occurred, use "Unclear".

    EVALUATION PRINCIPLES:
    1. Evidence over impression:
    - Every conclusion must be supported by something directly stated or clearly observable in the transcript.
    2. Fairness:
    - Consider complexity, customer behavior, and system limitations only if explicitly reflected in the transcript.
    3. Consistency:
    - Use conservative scoring when evidence is mixed or limited.
    4. Actionability:
    - Improvement areas should describe what the agent could do differently in future interactions.
    5. No over-penalization:
    - A minor wording issue, incomplete empathy phrase, or isolated imperfection is not automatically an infraction.

    INFRACTION DETECTION CRITERIA:
    1. Educational Gap:
    Mark only when the transcript shows a meaningful lack of product knowledge, policy understanding, or procedural awareness that affected or could reasonably affect the handling of the interaction.
    2. Unactioned Threat Detection:
    Mark only when the customer explicitly mentions or clearly threatens escalation to an external or high-risk party and the agent fails to respond appropriately.
    Relevant threat or escalation references include:
    - CRTC, CCTS
    - CEO, Office of the President, executive office
    - Media, reporters, social media exposure, public complaint
    - Lawyer, lawsuit, legal action, court
    - Police, law enforcement
    - Any regulator, ombudsman, government body, or external complaint authority

    ESCALATION ASSESSMENT:
    Assess all of the following based only on transcript evidence:
    - Whether manager/supervisor escalation occurred
    - Whether escalation was appropriate
    - Whether escalation timing was early, appropriate, late, never, or unclear
    - Whether escalation improved handling or resolution

    RATING GUIDELINES:
    Use the following standards consistently:

    communication_skills.rating:
    - "Poor" = unclear, inappropriate, confusing, or repeatedly ineffective communication
    - "Fair" = understandable but inconsistent, incomplete, or somewhat unclear communication
    - "Good" = generally clear, professional, and appropriate communication
    - "Excellent" = consistently clear, professional, precise, and well-managed communication

    empathy_level.rating:
    - "Low" = little or no acknowledgment of customer concerns
    - "Medium" = some acknowledgment, but limited or inconsistent
    - "High" = clear acknowledgment and appropriate emotional responsiveness

    professionalism_level.rating:
    - "Poor" = disrespectful, dismissive, defensive, or inappropriate
    - "Fair" = mostly professional but with notable lapses
    - "Good" = respectful and professional throughout most of the interaction
    - "Excellent" = consistently respectful, composed, accountable, and professional

    resolution_efficiency.rating:
    - "Inefficient" = repetitive, poorly directed, delayed, or lacking forward movement
    - "Average" = some progress but with avoidable repetition or lack of clarity
    - "Efficient" = reasonably organized and moved the issue forward appropriately
    - "Highly Efficient" = focused, well-paced, and advanced resolution with minimal unnecessary repetition

    HOW TO HANDLE LIMITED EVIDENCE & N/A USAGE:
    - If the transcript does not provide enough evidence for a category, use "N/A" or "Unclear" in the justification field.
    - Do not lower a rating solely because the transcript is short.
    - Do not assume empathy, escalation, or procedural failure unless it is observable.
    - If a threat or escalation reference is absent, set "unactioned_threat_detection" to "No".
    - If a threat is mentioned but the response cannot be evaluated due to missing context, set the relevant assessment to "Unclear" and explain briefly.
    
    WHEN TO USE "N/A" IN AGENT EVALUATION:
    - For justification fields: Use "N/A" when evidence is insufficient to provide a meaningful explanation
    - For educational_gap_detected: Use empty array [] if no gaps are evident; use "N/A" in educational_gap_details if not applicable
    - For key_strengths/improvement_areas: Use empty arrays [] if not supported by evidence
    - For threat_handling_assessment: Use "N/A" if no threat or escalation language appears in transcript

    ANALYSIS STEPS:
    1. Identify the target agent and any clearly attributable statements.
    2. Summarize overall performance using transcript evidence only.
    3. Identify strengths supported by observable examples.
    4. Identify improvement areas supported by observable examples.
    5. Assess whether any meaningful infraction occurred.
    6. Assess whether escalation occurred and whether it was appropriate.
    7. Return the result in valid JSON only.

    OUTPUT REQUIREMENTS:
    - Return ONLY valid JSON.
    - Do not include markdown fences.
    - Do not include commentary before or after the JSON.
    - Use exactly the field names and allowed values shown below.
    - All string values must be valid JSON strings.
    - Arrays must always be included, even if empty.
    - If no educational gap is detected, use an empty array for "educational_gap_detected" and "N/A" for "educational_gap_details".
    - If no strengths or improvement areas are supported by evidence, return an empty array for that field.
    - Do not add extra keys.
    - Do not omit required keys.

    REQUIRED OUTPUT JSON FORMAT:
    {{
    "agent_evaluations": [
        {{
        "agent_identifier": "{row['emp_id']}",
        "performance_evaluation": {{
            "overall_performance": "Evidence-based summary of the agent's performance. If evidence is limited, clearly state that.",
            "key_strengths": [
            "Specific positive behavior supported by transcript evidence"
            ],
            "improvement_areas": [
            "Specific agent improvement areas that can prevent CCTS complaints"
            ],
            "communication_skills": {{
            "rating": "Poor/Fair/Good/Excellent",
            "justification": "Evidence-based explanation. If evidence is limited, say so."
            }},
            "empathy_level": {{
            "rating": "Low/Medium/High",
            "justification": "Evidence-based explanation. If evidence is limited, say so."
            }},
            "professionalism_level": {{
            "rating": "Poor/Fair/Good/Excellent",
            "justification": "Evidence-based explanation. If evidence is limited, say so"
            }},
            "resolution_efficiency": {{
            "rating": "Inefficient/Average/Efficient/Highly Efficient",
            "justification": "Evidence-based explanation. If evidence is limited, say so."
            }},
            "evaluation_of_next_steps": "State whether the agent provided clear, appropriate, and actionable next steps, or state that evidence is insufficient."
        }},
        "infraction_assessment": {{
            "agent_infraction": "Yes/No",
            "infraction_rationale": "Neutral explanation based only on transcript evidence.",
            "educational_gap_detected": [
            "Meaningful knowledge or process gap supported, focus on how agent can do better to avoid complaint"
            ],
            "educational_gap_details": [
                "Specific evidence of knowledge or policy gaps. Point out where agent can improve to avoid CCTS complaint"
            ],
            "unactioned_threat_detection": "Yes/No",
            "threat_handling_assessment": "Describe whether any threat or escalation language was present, how the agent responded, and whether handling was appropriate. If no threat was present, say so."
        }},
        "internal_escalation": {{
            "escalation_appropriate": "Yes/No/Unclear",
            "escalation_timing": "Early/Appropriate/Late/Never/Unclear",
            "escalation_effectiveness": "State whether manager or supervisor involvement occurred and whether it helped, or explain why not applicable or unclear."
        }}
        }}
    ]
    }}
    """
    return agent_eval


def get_journey_analysis_prompt(results: list, file_number: str) -> str:
    """
    Generate the prompt for customer journey analysis.

    Args:
        results: List of analysis results
        file_number: File number for the case

    Returns:
        Formatted prompt string
    """
    import json

    journey_prompt = f"""You are a senior customer experience analyst conducting a comprehensive CCTS complaint journey and root cause analysis.
    Your task is to analyze all available customer interactions from the same account and determine how the complaint developed, why it escalated, where the company response failed, and what could have prevented the regulatory complaint.

    You have access to {len(results)} customer interactions associated with this account.

    ANALYSIS DATA:
    {json.dumps(results, indent=2)}
    END ANALYSIS DATA

    PRIMARY OBJECTIVE:
    Produce an evidence-based analysis of the customer journey that explains:
    1. what the complaint was fundamentally about,
    2. how the issue evolved over time,
    3. where the customer experience broke down,
    4. whether the company response matched the issue and customer impact,
    5. and what Rogers could do differently to prevent similar complaints.

    GENERAL RULES:
    - Use ONLY the interaction data provided.
    - Do not assume facts not stated in the data.
    - Do not speculate about intent, policy, internal systems, or missing actions unless supported by evidence.
    - If evidence is limited or conflicting, say so explicitly.
    - Prefer conservative conclusions over unsupported certainty.
    - Focus on observable events, promises, failures, delays, escalations, and customer reactions.
    - Identify patterns across interactions when supported by multiple records.
    - Distinguish between:
    - the original problem,
    - the factors that prolonged or worsened it,
    - the specific trigger for the CCTS complaint,
    - and the broader systemic contributors.

    EVIDENCE REQUIREMENTS:
    - Every major conclusion must be supported by at least one citation.
    - Use citation format exactly as follows:
    [Interaction Date/ID: specific_identifier - "relevant quote, action, promise, or behavior"]
    - If an interaction has no date, use the best available identifier.
    - If quoting is not possible, cite a concise evidence-based behavior description from the record.
    - Do not cite interactions that do not support the statement.
    - If evidence is insufficient, explicitly state "Insufficient evidence."

    ANALYSIS STANDARDS:
    - Verify that the summary of each issue is accurate and grounded in the source data.
    - Identify recurring patterns such as repeat contacts, repeated unmet commitments, inconsistent messaging, delayed resolution, billing disputes, service failures, compensation dissatisfaction, or escalation mishandling.
    - Separate customer perception from verifiable facts when they differ.
    - Evaluate both immediate failures and underlying process weaknesses.
    - Focus on actionable prevention opportunities.

    HOW TO HANDLE LIMITED EVIDENCE:
    - If the records do not show whether a promised action was completed, say "Insufficient evidence to confirm completion."
    - If customer expectations are unclear, describe only what is explicitly stated.
    - If the final trigger for the complaint cannot be confidently isolated, identify the most likely trigger and state the limitation.
    - Do not invent missing chronology.
    - If no clear systemic issue is supported, say so.

    WHEN TO USE "N/A" OR "INSUFFICIENT EVIDENCE" IN JOURNEY ANALYSIS:
    - primary_complaint_issue: Use "Insufficient evidence" only if interactions contain no clear complaint-related content; otherwise always derive from data
    - issue_evolution: Use "Insufficient evidence" if only one interaction exists or if evolution cannot be traced; otherwise describe what the data shows
    - unresolved_issues: Use empty array [] if all issues show resolution; use specific issues with citations; use ["Insufficient evidence"] if resolution status is unclear
    - solutions_offered: Use [] if no remedies, credits, commitments, or explanations were offered; never leave empty without justification
    - implementation_gaps: Use [] if no promised actions failed or delayed; use specific gaps with citations if any delays or failures appear in records
    - consistency_of_handling: Use "Insufficient evidence" only if interactions are too limited to assess consistency; otherwise describe consistency or lack thereof
    - critical_breakdown_moments: Use [] if customer experience remained stable; use specific moments with citations if deterioration is evident
    - repeat_contact_pattern: Use "Insufficient evidence" if unclear why customer repeated contact; otherwise explicitly state pattern findings with citations
    - final_straw_incident: Use "Insufficient evidence to identify final trigger" if complaint trigger cannot be confidently isolated; otherwise cite the specific moment
    - proactive_outreach: Use [] if no outreach opportunities are evident; otherwise list with citations
    - compensation_timing: Use [] if no compensation was offered or timing was not problematic; otherwise list opportunities with citations
    - escalation_management: Use [] if escalation was handled appropriately or was not a factor; otherwise list management failures with citations
    - strategic_recommendations: Never empty; always provide at least 3-5 specific operational recommendations based on analysis

    OUTPUT REQUIREMENTS:
    - Return ONLY valid JSON.
    - Do not include markdown fences.
    - Do not include any text before or after the JSON object.
    - Use exactly the keys shown below.
    - Do not add extra keys.
    - Do not omit required keys.
    - All arrays must always be included, even if empty.
    - All required text fields must be filled. If evidence is insufficient, use "Insufficient evidence."
    - Keep explanations concise but specific.
    - All major conclusions should include at least one embedded citation in the text or be supported in a related evidence array.

    REQUIRED OUTPUT JSON FORMAT:
    {{
    "ccts_complaint_journey_analysis": {{
        "case_number": "{file_number}",
        "customer_complaint_genesis": {{
        "primary_complaint_issue": "State the core issue that prompted the CCTS complaint, with brief explanation and citation(s).",
        "issue_evolution": "Explain how the problem developed over time across interactions, including major turning points, with citation(s).",
        "unresolved_issues": [
            "List unresolved issues that remained open or inadequately resolved, each with citation(s)."
        ]
        }},
        "response_assessment": {{
        "solutions_offered": [
            "List specific remedies, credits, explanations, commitments, or resolutions offered by the company, each with citation(s)."
        ],
        "implementation_gaps": [
            "List promised actions or resolutions that were delayed, not completed, unclear, or inconsistently applied, each with citation(s)."
        ],
        "consistency_of_handling": "Assess whether the customer received consistent or inconsistent handling across interactions, with citation(s)."
        }},
        "journey_failure_points": {{
        "critical_breakdown_moments": [
            "Identify specific interactions where trust, clarity, or resolution materially deteriorated, with citation(s)."
        ],
        "repeat_contact_pattern": "State whether repeat contacts indicate unresolved handling, with citation(s), or 'Insufficient evidence.'",
        "final_straw_incident": "Identify the specific interaction or event that most directly triggered the CCTS complaint, with citation(s), or explain if uncertain."
        }}
    }},
    "value_gap_analysis": {{
        "offer_vs_expectation_matrix": "Please provide a detailed value gap analysis e.g., monetary, expectation, communication, and timing. please provide detailed comparison between offer and expectation",
        "rationality_assessment": {{
        "customer_demand_rationality": "Reasonable/Partially Reasonable/Unreasonable",
        "customer_demand_rationality_justification": "Explain whether the customer's demands were proportionate to the issue and impact, with citation(s).",
        "company_offer_adequacy": "Adequate/Inadequate/Partially Adequate",
        "company_offer_adequacy_justification": "Explain whether the company's response matched the issue severity and customer impact, with citation(s)."
        }}
    }},
    "prevention_opportunity_analysis": {{
        "proactive_outreach": [
        "Identify opportunities where the company could have contacted or reassured the customer earlier to prevent escalation, with citation(s)."
        ],
        "compensation_timing": [
        "Identify whether compensation could have been offered earlier, more clearly, or more appropriately, with citation(s)."
        ],
        "escalation_management": [
        "Identify where stronger internal escalation, ownership, or callback management could have prevented external escalation, with citation(s)."
        ]
    }},
    "resolution_recommendations": {{
        "root_cause_identification": {{
        "primary_root_cause": "State the fundamental issue that created or sustained the complaint pattern, with citation(s).",
        "contributing_factors": [
            "List secondary issues that amplified the complaint, each with citation(s)."
        ],
        "systemic_vs_individual": "State whether the issue appears systemic, individual, mixed, or unclear, with brief justification and citation(s).",
        "cause_explanation": "Explain why these factors caused escalation to a regulatory complaint, with citation(s).",
        "evidence_base": [
            "List the strongest supporting citations for the root cause analysis."
        ]
        }},
        "strategic_recommendations": [
        "List 3-4 resolution recommendations from operational perspective to prevent CCTS complaint escalation"
        ]
    }}
    }}

    IMPORTANT DECISION RULES:
    - Do not label something a root cause unless multiple facts support it or it clearly drove the complaint.
    - Do not confuse a symptom with the root cause.
    - Do not assume a failed promise unless the data shows the promise was unmet, contradicted, or unconfirmed.
    - If there were multiple issues, identify the main issue and distinguish secondary ones.
    - If the company made reasonable efforts but resolution still failed, state that fairly.
    - If the customer expectation exceeded what the evidence supports, note that neutrally.
    - When in doubt, choose precise and evidence-based wording over strong language.

    IMPORTANT:
    Return ONLY valid JSON. Do not include any text before or after the JSON object.
    """
    return journey_prompt
