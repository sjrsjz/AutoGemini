"""
Interactive streaming chat test for Gemini API.
This is a CLI-based test that allows real interaction with the Gemini API.
"""

import asyncio
import os
import sys
from typing import List, Optional
from datetime import datetime

from autogemini import stream_chat, ChatMessage, MessageRole, StreamCancellation, fetch_available_models


class InteractiveChatTester:
    """Interactive chat tester with CLI interface."""
    
    def __init__(self):
        self.api_key: Optional[str] = None
        self.history: List[ChatMessage] = []
        self.current_model = "gemini-2.0-flash-thinking-exp"
        self.system_prompt = "You are a helpful AI assistant."
        self.temperature = 0.7
        self.max_tokens = 8192
        self.cancellation: Optional[StreamCancellation] = None
        
    def setup_api_key(self):
        """Setup API key from environment or user input."""
        # Try to get from environment first
        self.api_key = os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            print("🔑 Gemini API key not found in environment variable 'GEMINI_API_KEY'")
            self.api_key = input("Please enter your Gemini API key: ").strip()
            
            if not self.api_key:
                print("❌ No API key provided. Exiting.")
                sys.exit(1)
        
        print(f"✅ API key loaded (ending with ...{self.api_key[-4:]})")
    
    def print_header(self):
        """Print welcome header."""
        print("="*60)
        print("🤖 Interactive Gemini Streaming Chat Tester")
        print("="*60)
        print("Commands:")
        print("  /help     - Show help")
        print("  /models   - List available models")
        print("  /model    - Change model")
        print("  /system   - Change system prompt")
        print("  /temp     - Change temperature")
        print("  /tokens   - Change max tokens")
        print("  /history  - Show conversation history")
        print("  /clear    - Clear conversation history")
        print("  /cancel   - Cancel current stream (during streaming)")
        print("  /quit     - Exit the chat")
        print("-"*60)
        print(f"Current model: {self.current_model}")
        print(f"System prompt: {self.system_prompt}")
        print(f"Temperature: {self.temperature}")
        print(f"Max tokens: {self.max_tokens}")
        print("="*60)
    
    async def handle_models_command(self):
        """Handle /models command."""
        if not self.api_key:
            print("❌ No API key available")
            return
            
        print("🔍 Fetching available models...")
        try:
            models = await fetch_available_models(self.api_key)
            print(f"\n📋 Available Gemini models ({len(models)} total):")
            for i, model in enumerate(models, 1):
                marker = "👉" if model == self.current_model else "  "
                print(f"{marker} {i:2d}. {model}")
        except Exception as e:
            print(f"❌ Error fetching models: {e}")
    
    def handle_model_command(self):
        """Handle /model command."""
        print(f"Current model: {self.current_model}")
        new_model = input("Enter new model name (or press Enter to keep current): ").strip()
        if new_model:
            self.current_model = new_model
            print(f"✅ Model changed to: {self.current_model}")
    
    def handle_system_command(self):
        """Handle /system command."""
        print(f"Current system prompt: {self.system_prompt}")
        new_prompt = input("Enter new system prompt (or press Enter to keep current): ").strip()
        if new_prompt:
            self.system_prompt = new_prompt
            print("✅ System prompt updated")
    
    def handle_temp_command(self):
        """Handle /temp command."""
        print(f"Current temperature: {self.temperature}")
        try:
            temp_input = input("Enter new temperature (0.0-1.0, or press Enter to keep current): ").strip()
            if temp_input:
                new_temp = float(temp_input)
                if 0.0 <= new_temp <= 1.0:
                    self.temperature = new_temp
                    print(f"✅ Temperature changed to: {self.temperature}")
                else:
                    print("❌ Temperature must be between 0.0 and 1.0")
        except ValueError:
            print("❌ Invalid temperature value")
    
    def handle_tokens_command(self):
        """Handle /tokens command."""
        print(f"Current max tokens: {self.max_tokens}")
        try:
            tokens_input = input("Enter new max tokens (or press Enter to keep current): ").strip()
            if tokens_input:
                new_tokens = int(tokens_input)
                if new_tokens > 0:
                    self.max_tokens = new_tokens
                    print(f"✅ Max tokens changed to: {self.max_tokens}")
                else:
                    print("❌ Max tokens must be positive")
        except ValueError:
            print("❌ Invalid max tokens value")
    
    def handle_history_command(self):
        """Handle /history command."""
        if not self.history:
            print("📝 No conversation history")
            return
        
        print(f"📚 Conversation History ({len(self.history)} messages):")
        print("-" * 50)
        
        for i, msg in enumerate(self.history, 1):
            role_icon = "👤" if msg.role == MessageRole.USER else "🤖"
            role_name = msg.role.value.upper()
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            print(f"{role_icon} [{timestamp}] {role_name}:")
            
            # Truncate long messages for display
            content = msg.content
            if len(content) > 200:
                content = content[:200] + "..."
            
            print(f"   {content}")
            print()
    
    def handle_clear_command(self):
        """Handle /clear command."""
        if self.history:
            confirm = input(f"Are you sure you want to clear {len(self.history)} messages? (y/N): ").strip().lower()
            if confirm == 'y':
                self.history.clear()
                print("✅ Conversation history cleared")
            else:
                print("❌ Clear cancelled")
        else:
            print("📝 History is already empty")
    
    def handle_help_command(self):
        """Handle /help command."""
        print("\n📖 Help:")
        print("  • Type your message and press Enter to chat")
        print("  • Use commands starting with '/' for configuration")
        print("  • During streaming, you can type '/cancel' to stop")
        print("  • The AI response will stream in real-time")
        print("  • Conversation history is maintained automatically")
        print("  • Use Ctrl+C to interrupt or exit")
    
    def handle_cancel_command(self):
        """Handle /cancel command."""
        if self.cancellation and not self.cancellation.is_cancelled():
            self.cancellation.cancel()
            print("\n🛑 Stream cancellation requested...")
        else:
            print("❌ No active stream to cancel")
    
    def on_stream_chunk(self, chunk: str):
        """Callback for streaming chunks."""
        print(chunk, end='', flush=True)
        print("[...]", end='', flush=True)  # Indicate streaming in progress
    
    async def send_message(self, user_message: str) -> bool:
        """Send a message and handle streaming response."""
        if not self.api_key:
            print("❌ No API key available")
            return False
            
        print(f"\n👤 You: {user_message}")
        print("🤖 Assistant: ", end='', flush=True)
        
        # Create cancellation token
        self.cancellation = StreamCancellation()
        
        try:
            # Send message with streaming
            response = await stream_chat(
                api_key=self.api_key,
                user_message=user_message,
                callback=self.on_stream_chunk,
                history=self.history,
                model=self.current_model,
                system_prompt=self.system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                cancellation_token=self.cancellation,
                timeout=300.0
            )
            
            print()  # New line after response
            
            # Add to history if not cancelled
            if not self.cancellation.is_cancelled():
                self.history.append(ChatMessage(role=MessageRole.USER, content=user_message))
                self.history.append(ChatMessage(role=MessageRole.ASSISTANT, content=response))
                print(f"💾 Messages saved to history (total: {len(self.history)})")
            else:
                print("🛑 Stream was cancelled - messages not saved to history")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return False
        finally:
            self.cancellation = None
    
    async def run(self):
        """Run the interactive chat."""
        self.setup_api_key()
        self.print_header()
        
        print("\n💬 Ready to chat! Type your message or use commands.")
        print("   Type '/quit' to exit.\n")
        
        while True:
            try:
                # Get user input
                user_input = input(">>> ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    command = user_input.lower()
                    
                    if command == '/quit':
                        print("👋 Goodbye!")
                        break
                    elif command == '/help':
                        self.handle_help_command()
                    elif command == '/models':
                        await self.handle_models_command()
                    elif command == '/model':
                        self.handle_model_command()
                    elif command == '/system':
                        self.handle_system_command()
                    elif command == '/temp':
                        self.handle_temp_command()
                    elif command == '/tokens':
                        self.handle_tokens_command()
                    elif command == '/history':
                        self.handle_history_command()
                    elif command == '/clear':
                        self.handle_clear_command()
                    elif command == '/cancel':
                        self.handle_cancel_command()
                    else:
                        print(f"❌ Unknown command: {user_input}")
                        print("   Type '/help' for available commands")
                
                else:
                    # Send regular message
                    await self.send_message(user_input)
                
                print()  # Extra spacing
                
            except KeyboardInterrupt:
                if self.cancellation and not self.cancellation.is_cancelled():
                    print("\n🛑 Cancelling current stream...")
                    self.cancellation.cancel()
                    await asyncio.sleep(0.1)  # Give time for cancellation
                else:
                    print("\n\n👋 Interrupted. Goodbye!")
                    break
            except EOFError:
                print("\n\n👋 EOF detected. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                continue


async def main():
    """Main test function."""
    tester = InteractiveChatTester()
    await tester.run()


if __name__ == "__main__":
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
