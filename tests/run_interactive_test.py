#!/usr/bin/env python3
"""
Run the interactive chat test.
Usage: python tests/run_interactive_test.py
"""

import asyncio
import sys

from test_stream_chat import main

if __name__ == "__main__":
    print("🚀 Starting Interactive Gemini Chat Test...")
    
    # Set up proper event loop policy for Windows
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Test interrupted. Goodbye!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
