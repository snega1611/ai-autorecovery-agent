from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine


POLICY_NUMBER_PATTERN = Pattern(
    name="policy_number_pattern",
    regex=r"\bP\d{4,6}\b",
    score=0.95,
)

policy_number_recognizer = PatternRecognizer(
    supported_entity="POLICY_NUMBER",
    patterns=[POLICY_NUMBER_PATTERN],
)


ANALYZED_ENTITIES = [
    "POLICY_NUMBER",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "PERSON",
    "LOCATION",
]


def sanitize_for_llm(text: str) -> str:
    analyzer = AnalyzerEngine()
    analyzer.registry.add_recognizer(policy_number_recognizer)

    results = analyzer.analyze(
        text=text,
        language="en",
        entities=ANALYZED_ENTITIES,
    )

    # POLICY_NUMBER is required for Recall's recovery workflow,
    # so remove it from the anonymization candidates.
    pii_to_mask = [
        result
        for result in results
        if result.entity_type != "POLICY_NUMBER"
    ]

    anonymizer = AnonymizerEngine()

    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=pii_to_mask,
    )

    return anonymized.text