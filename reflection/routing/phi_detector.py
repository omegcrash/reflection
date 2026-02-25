# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
PHI/PII Detection for HIPAA Compliance

Detects Protected Health Information and Personally Identifiable Information
to enable automatic routing to compliant infrastructure.
"""

import re


class PHIDetector:
    """
    Detect Protected Health Information (HIPAA 18 identifiers).

    PHI includes:
    - Medical Record Numbers (MRN)
    - Social Security Numbers (SSN) in medical context
    - Dates of Birth with medical context
    - Diagnosis codes (ICD-10)
    - Medication prescriptions
    - Health insurance information
    - Medical procedures (CPT codes)
    - Lab results
    """

    # PHI patterns (HIPAA 18 identifiers)
    PATTERNS = {
        "mrn": r"\b(MRN|Medical Record Number|Patient ID):\s*\d+\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "dob_context": r"\b(DOB|Date of Birth|Born|Birth Date):\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        "diagnosis": r"\b(ICD-10|ICD10|Diagnosis|Dx):\s*[A-Z]\d{2}\.?\d*\b",
        "medication_rx": r"\b(Rx|Prescription|Medication|Drug):\s*[A-Z][a-z]+\b",
        "health_insurance": r"\b(Insurance|Policy Number|Member ID|Subscriber ID):\s*[A-Z0-9-]+\b",
        "medical_procedure": r"\b(CPT|Procedure Code):\s*\d{5}\b",
        "lab_results": r"\b(Lab|Test Result|Blood|Glucose|A1C|Hemoglobin):\s*\d+\.?\d*\b",
        "vital_signs": r"\b(BP|Blood Pressure|HR|Heart Rate|Temp|Temperature):\s*\d+[/\d]*\b",
    }

    # Medical context keywords
    MEDICAL_KEYWORDS = [
        "patient",
        "diagnosis",
        "treatment",
        "medication",
        "prescription",
        "doctor",
        "physician",
        "nurse",
        "hospital",
        "clinic",
        "medical",
        "surgery",
        "operation",
        "exam",
        "examination",
        "test",
        "lab",
        "result",
        "symptom",
        "pain",
        "fever",
        "infection",
        "disease",
        "condition",
        "therapy",
        "drug",
        "dose",
        "dosage",
        "allergy",
        "allergies",
        "vital signs",
        "vitals",
        "chart",
        "ehr",
        "emr",
        "medical record",
    ]

    def contains_phi(self, text: str) -> tuple[bool, list[str]]:
        """
        Check if text contains PHI.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (contains_phi, detected_types)

        Examples:
            >>> detector = PHIDetector()
            >>> detector.contains_phi("Patient MRN: 12345")
            (True, ['mrn'])
            >>> detector.contains_phi("Hello world")
            (False, [])
        """
        detected_types = []

        # Check explicit PHI patterns
        for phi_type, pattern in self.PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                detected_types.append(phi_type)

        # Check for medical context + PII
        # If we have medical keywords + SSN/DOB, it's likely PHI
        text_lower = text.lower()
        medical_keyword_count = sum(1 for keyword in self.MEDICAL_KEYWORDS if keyword in text_lower)

        if medical_keyword_count >= 2:
            # Check for SSN or DOB in medical context
            if re.search(r"\b\d{3}-\d{2}-\d{4}\b", text) and "ssn" not in detected_types:
                detected_types.append("ssn_medical_context")

            if (
                re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text)
                and "dob_context" not in detected_types
            ):
                detected_types.append("dob_medical_context")

        return (len(detected_types) > 0, detected_types)


class PIIDetector:
    """
    Detect Personally Identifiable Information.

    PII that's not necessarily PHI:
    - Email addresses
    - Phone numbers
    - Physical addresses
    - Credit card numbers
    - SSN (without medical context)
    """

    PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
        "address": r"\b\d+\s+[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}(-\d{4})?\b",
        "zip_code": r"\b\d{5}(-\d{4})?\b",
    }

    def contains_pii(self, text: str) -> tuple[bool, list[str]]:
        """
        Check if text contains PII.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (contains_pii, detected_types)
        """
        detected_types = []

        for pii_type, pattern in self.PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                detected_types.append(pii_type)

        return (len(detected_types) > 0, detected_types)
