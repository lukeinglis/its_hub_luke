#!/usr/bin/env python3
"""
Helper script to check which Claude models are available in your Vertex AI project.
"""

import os
from dotenv import load_dotenv
from anthropic import AnthropicVertex

# Load environment variables
load_dotenv()

project_id = os.getenv("VERTEX_PROJECT", "your-gcp-project-id")
location = os.getenv("VERTEX_LOCATION", "us-east5")

print(f"Checking Claude models in Vertex AI")
print(f"Project: {project_id}")
print(f"Location: {location}")
print("-" * 60)

# List of Claude models to test
models_to_test = [
    "claude-3-5-sonnet-v2@20241022",
    "claude-3-5-sonnet@20240620",
    "claude-3-opus@20240229",
    "claude-3-5-haiku@20241022",
    "claude-3-haiku@20240307",
]

print("\nTesting model availability:")
print()

for model_name in models_to_test:
    try:
        client = AnthropicVertex(
            project_id=project_id,
            region=location,
        )

        # Try a minimal request to check if model exists
        response = client.messages.create(
            model=model_name,
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}]
        )

        print(f"✓ {model_name:<40} AVAILABLE")

    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "NOT_FOUND" in error_msg:
            print(f"✗ {model_name:<40} NOT FOUND")
        elif "403" in error_msg or "PERMISSION_DENIED" in error_msg:
            print(f"⚠ {model_name:<40} NO PERMISSION")
        else:
            print(f"? {model_name:<40} ERROR: {error_msg[:50]}")

print()
print("-" * 60)
print("\nTip: If no models are available, you may need to:")
print("1. Enable Vertex AI API: gcloud services enable aiplatform.googleapis.com")
print("2. Request access to Claude models in Google Cloud Console")
print("3. Check that you're authenticated: gcloud auth application-default login")
