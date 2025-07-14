#!/usr/bin/env python3
"""
Simple test script to verify the Multi-Agent RAG System API is working correctly.
Run this script after deployment to ensure everything is functioning.
"""

import requests
import json
import time
import sys

# Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 30

def test_health_check():
    """Test the health check endpoint."""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Health check passed: {data['status']}")
            print(f"  PDF service: {data['pdf_service']}")
            print(f"  Papers loaded: {data['papers_loaded']}")
            print(f"  Agent system: {data['agent_system']}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint."""
    print("\nTesting root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Root endpoint working: {data['message']}")
            return True
        else:
            print(f"✗ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Root endpoint error: {e}")
        return False

def test_session_creation():
    """Test session creation."""
    print("\nTesting session creation...")
    try:
        response = requests.post(f"{BASE_URL}/session", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            session_id = data['session_id']
            print(f"✓ Session created: {session_id}")
            return session_id
        else:
            print(f"✗ Session creation failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"✗ Session creation error: {e}")
        return None

def test_ask_question(session_id=None):
    """Test asking a question."""
    print("\nTesting question asking...")
    
    test_questions = [
        "What papers are available?",
        "List the research papers in the database",
    ]
    
    for question in test_questions:
        try:
            payload = {"question": question}
            if session_id:
                payload["session_id"] = session_id
            
            print(f"  Asking: {question}")
            response = requests.post(
                f"{BASE_URL}/ask", 
                json=payload, 
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"  ✓ Response type: {data['type']}")
                print(f"  ✓ Session: {data['session_id']}")
                if data.get('search_strategy'):
                    print(f"  ✓ Strategy: {data['search_strategy']}")
                print(f"  ✓ Response length: {len(data['response'])} chars")
                return True
            else:
                print(f"  ✗ Question failed: {response.status_code}")
                print(f"  Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"  ✗ Question error: {e}")
            return False
    
    return True

def test_session_history(session_id):
    """Test getting session history."""
    if not session_id:
        print("\nSkipping session history test (no session ID)")
        return True
    
    print(f"\nTesting session history for {session_id}...")
    try:
        response = requests.get(f"{BASE_URL}/session/{session_id}/history", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Session history retrieved: {len(data['history'])} items")
            return True
        else:
            print(f"✗ Session history failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Session history error: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Multi-Agent RAG System API Tests ===\n")
    
    # Wait a moment for the service to be ready
    print("Waiting for service to be ready...")
    time.sleep(5)
    
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Health check
    if test_health_check():
        tests_passed += 1
    
    # Test 2: Root endpoint
    if test_root_endpoint():
        tests_passed += 1
    
    # Test 3: Session creation
    session_id = test_session_creation()
    if session_id:
        tests_passed += 1
    
    # Test 4: Ask question
    if test_ask_question(session_id):
        tests_passed += 1
    
    # Test 5: Session history
    if test_session_history(session_id):
        tests_passed += 1
    
    # Summary
    print(f"\n=== Test Results ===")
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! API is working correctly.")
        sys.exit(0)
    else:
        print("✗ Some tests failed. Check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 