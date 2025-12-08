# stem.py
import os
import sys
from google import genai
from google.genai.errors import APIError

def test_gemini_connection():
    """
    Tests connection to the Gemini API using the GEMINI_API_KEY environment variable.
    """
    # The client automatically looks for the GEMINI_API_KEY environment variable.
    try:
        # Check if the API key environment variable is set
        if "GEMINI_API_KEY" not in os.environ:
            print("❌ FAILURE: GEMINI_API_KEY environment variable not found.")
            sys.exit(1)

        print("✅ API Key found in environment variables.")
        client = genai.Client()
        
        # Simple request to test connectivity
        print("Sending test request to Gemini...")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Say 'Hello from GitHub Actions!'",
        )

        # Check for a valid response
        if response.text and len(response.text.strip()) > 0:
            print("--- API Response ---")
            print(response.text.strip())
            print("--------------------")
            print("✅ SUCCESS: Successfully connected to Gemini API and received a response.")
            sys.exit(0)
        else:
            print("❌ FAILURE: Connected but received an empty or unreadable response.")
            sys.exit(1)

    except APIError as e:
        print(f"❌ FAILURE: Gemini API Error occurred: {e}")
        # An API error often means the key is invalid or permissions are wrong
        sys.exit(1)
    except Exception as e:
        print(f"❌ FAILURE: An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_gemini_connection()
