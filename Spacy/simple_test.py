#!/usr/bin/env python3
"""
Simple test of the improved custom mode function
"""

print("Testing import...")
from data_generation_noisy import generate_example_with_custom_mode
print("Import successful!")

print("\nTesting function...")
sentence, annotations = generate_example_with_custom_mode(country='chile', mode='personal_id')
print(f"Generated: {sentence}")
print(f"Entities: {annotations}")

# Check for failed spans
entities = annotations.get('entities', [])
failed_spans = 0
for start, end, label in entities:
    if start >= end or start < 0 or end > len(sentence):
        failed_spans += 1
        print(f"FAILED SPAN: {start}:{end} for {label}")
    else:
        entity_text = sentence[start:end]
        if not entity_text.strip():
            failed_spans += 1
            print(f"EMPTY ENTITY: {start}:{end} for {label}")

print(f"\nFailed spans: {failed_spans}")
print("Test completed successfully!")