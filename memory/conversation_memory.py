import json
import os
import logging
import time
from collections import deque
import sqlite3
import config

logger = logging.getLogger(__name__)

class ConversationMemory:
    """System for maintaining conversation memory between users and agents"""
    
    def __init__(self, memory_type=None, max_tokens=None, db_path=None):
        self.memory_type = memory_type or config.MEMORY_TYPE
        self.max_tokens = max_tokens or config.MEMORY_MAX_TOKENS
        
        # Set up the appropriate memory storage based on the memory type
        if self.memory_type == "buffer":
            self.memory_store = BufferMemory(max_tokens=self.max_tokens)
        elif self.memory_type == "summary":
            self.memory_store = SummaryMemory(max_tokens=self.max_tokens)
        elif self.memory_type == "database":
            db_path = db_path or os.path.join(os.getcwd(), "memory", "conversations.db")
            self.memory_store = DatabaseMemory(db_path=db_path)
        else:
            logger.warning(f"Unknown memory type: {self.memory_type}. Using buffer memory.")
            self.memory_store = BufferMemory(max_tokens=self.max_tokens)
    
    def add_exchange(self, user_message, agent_response):
        """Add a new exchange to memory"""
        self.memory_store.add_exchange(user_message, agent_response)
    
    def update_memory(self, conversation_history):
        """Update memory with the full conversation history"""
        self.memory_store.clear()
        for user_msg, agent_resp in conversation_history:
            self.memory_store.add_exchange(user_msg, agent_resp)
    
    def get_context(self):
        """Get the memory context for agents"""
        return self.memory_store.get_context()
    
    def clear(self):
        """Clear the memory"""
        self.memory_store.clear()


class BufferMemory:
    """Simple buffer-based memory that stores full conversation history up to a token limit"""
    
    def __init__(self, max_tokens=8000):
        self.max_tokens = max_tokens
        self.exchanges = deque()
        self.current_token_count = 0
    
    def add_exchange(self, user_message, agent_response):
        """Add a new exchange to the buffer"""
        # Create exchange record
        exchange = {
            "user": user_message,
            "agent": agent_response,
            "timestamp": time.time(),
            "tokens": self._estimate_tokens(user_message + agent_response)
        }
        
        # Add to buffer
        self.exchanges.append(exchange)
        self.current_token_count += exchange["tokens"]
        
        # Trim buffer if needed
        self._trim_buffer()
    
    def get_context(self):
        """Get the conversation context from memory"""
        if not self.exchanges:
            return "No conversation history."
        
        context_parts = []
        for i, exchange in enumerate(self.exchanges):
            context_parts.append(f"Exchange {i+1}:")
            context_parts.append(f"User: {exchange['user']}")
            context_parts.append(f"Assistant: {exchange['agent']}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def clear(self):
        """Clear the memory buffer"""
        self.exchanges.clear()
        self.current_token_count = 0
    
    def _estimate_tokens(self, text):
        """Estimate the number of tokens in a text"""
        # Simple estimation: ~4 characters per token
        return len(text) // 4
    
    def _trim_buffer(self):
        """Trim the buffer to stay within token limit"""
        while self.current_token_count > self.max_tokens and self.exchanges:
            removed = self.exchanges.popleft()
            self.current_token_count -= removed["tokens"]


class SummaryMemory:
    """Memory that maintains a summary of the conversation plus recent exchanges"""
    
    def __init__(self, max_tokens=8000, summary_interval=5):
        self.max_tokens = max_tokens
        self.summary_interval = summary_interval
        self.exchanges = deque()
        self.current_token_count = 0
        self.summary = ""
        self.exchange_count = 0
    
    def add_exchange(self, user_message, agent_response):
        """Add a new exchange and update summary if needed"""
        # Create exchange record
        exchange = {
            "user": user_message,
            "agent": agent_response,
            "timestamp": time.time(),
            "tokens": self._estimate_tokens(user_message + agent_response)
        }
        
        # Add to buffer
        self.exchanges.append(exchange)
        self.current_token_count += exchange["tokens"]
        self.exchange_count += 1
        
        # Create or update summary if needed
        if self.exchange_count % self.summary_interval == 0:
            self._update_summary()
        
        # Trim buffer if needed
        self._trim_buffer()
    
    def get_context(self):
        """Get the conversation context with summary"""
        context_parts = []
        
        # Add summary if available
        if self.summary:
            context_parts.append("# Conversation Summary")
            context_parts.append(self.summary)
            context_parts.append("\n# Recent Exchanges")
        
        # Add recent exchanges
        for i, exchange in enumerate(self.exchanges):
            context_parts.append(f"Exchange {i+1}:")
            context_parts.append(f"User: {exchange['user']}")
            context_parts.append(f"Assistant: {exchange['agent']}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def clear(self):
        """Clear the memory"""
        self.exchanges.clear()
        self.current_token_count = 0
        self.summary = ""
        self.exchange_count = 0
    
    def _update_summary(self):
        """Update the conversation summary"""
        # This is a placeholder for an actual summarization function
        # In a real implementation, you would use an LLM to create the summary
        
        exchanges_text = ""
        for exchange in self.exchanges:
            exchanges_text += f"User: {exchange['user']}\n"
            exchanges_text += f"Assistant: {exchange['agent']}\n\n"
        
        # Simple summary (in production, use an LLM here)
        topics = self._extract_topics(exchanges_text)
        self.summary = f"This conversation covers topics including {', '.join(topics)}. "
        self.summary += f"There have been {self.exchange_count} total exchanges."
    
    def _extract_topics(self, text):
        """Extract topics from conversation (placeholder)"""
        # This is a very simple placeholder for topic extraction
        # In a real implementation, use NLP or an LLM for this
        topics = ["data analysis", "research", "statistics"]
        return topics
    
    def _estimate_tokens(self, text):
        """Estimate the number of tokens in a text"""
        # Simple estimation: ~4 characters per token
        return len(text) // 4
    
    def _trim_buffer(self):
        """Trim the buffer to stay within token limit"""
        summary_tokens = self._estimate_tokens(self.summary)
        max_exchange_tokens = self.max_tokens - summary_tokens
        
        while self.current_token_count > max_exchange_tokens and self.exchanges:
            removed = self.exchanges.popleft()
            self.current_token_count -= removed["tokens"]


class DatabaseMemory:
    """Memory system that stores conversations in a SQLite database"""
    
    def __init__(self, db_path, max_recent_exchanges=10):
        self.db_path = db_path
        self.max_recent_exchanges = max_recent_exchanges
        self.session_id = str(int(time.time()))  # Simple session ID based on timestamp
        
        # Create database directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize the database
        self._init_db()
        
        # Keep recent exchanges in memory for quick access
        self.recent_exchanges = deque(maxlen=max_recent_exchanges)
    
    def _init_db(self):
        """Initialize the database schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create sessions table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
                """)
                
                # Create exchanges table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS exchanges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    user_message TEXT,
                    agent_response TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
                """)
                
                # Create the current session
                cursor.execute("""
                INSERT OR IGNORE INTO sessions (session_id, metadata)
                VALUES (?, ?)
                """, (self.session_id, json.dumps({"source": "research_ai"})))
                
                conn.commit()
        
        except Exception as e:
            logger.error(f"Database initialization error: {str(e)}")
            # Fall back to in-memory operation if DB fails
            self.db_path = ":memory:"
            logger.warning(f"Using in-memory database due to initialization error")
            self._init_db()
    
    def add_exchange(self, user_message, agent_response):
        """Add a new exchange to the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Update session last_updated timestamp
                cursor.execute("""
                UPDATE sessions 
                SET last_updated = CURRENT_TIMESTAMP 
                WHERE session_id = ?
                """, (self.session_id,))
                
                # Insert the exchange
                cursor.execute("""
                INSERT INTO exchanges (session_id, user_message, agent_response)
                VALUES (?, ?, ?)
                """, (self.session_id, user_message, agent_response))
                
                conn.commit()
            
            # Add to recent exchanges deque
            self.recent_exchanges.append({
                "user": user_message,
                "agent": agent_response,
                "timestamp": time.time()
            })
        
        except Exception as e:
            logger.error(f"Error adding exchange to database: {str(e)}")
            # Still add to recent exchanges even if DB fails
            self.recent_exchanges.append({
                "user": user_message,
                "agent": agent_response,
                "timestamp": time.time()
            })
    
    def get_context(self):
        """Get conversation context from memory"""
        try:
            # Get recent exchanges from database if recent_exchanges is empty
            if not self.recent_exchanges:
                self._load_recent_exchanges()
            
            # Format the context
            context_parts = []
            
            # Add session info
            context_parts.append(f"Session ID: {self.session_id}")
            
            # Add recent exchanges
            for i, exchange in enumerate(self.recent_exchanges):
                context_parts.append(f"Exchange {i+1}:")
                context_parts.append(f"User: {exchange['user']}")
                context_parts.append(f"Assistant: {exchange['agent']}")
                context_parts.append("")
            
            return "\n".join(context_parts)
        
        except Exception as e:
            logger.error(f"Error getting context from database: {str(e)}")
            return "Error retrieving conversation history."
    
    def _load_recent_exchanges(self):
        """Load recent exchanges from the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get recent exchanges for the current session
                cursor.execute("""
                SELECT user_message, agent_response, timestamp
                FROM exchanges
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """, (self.session_id, self.max_recent_exchanges))
                
                rows = cursor.fetchall()
                
                # Add to recent exchanges deque (in chronological order)
                for user_msg, agent_resp, timestamp in reversed(rows):
                    self.recent_exchanges.append({
                        "user": user_msg,
                        "agent": agent_resp,
                        "timestamp": timestamp
                    })
        
        except Exception as e:
            logger.error(f"Error loading exchanges from database: {str(e)}")
    
    def clear(self):
        """Clear recent exchanges from memory but keep in database"""
        self.recent_exchanges.clear()
    
    def new_session(self):
        """Start a new session"""
        self.session_id = str(int(time.time()))
        self.recent_exchanges.clear()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create the new session
                cursor.execute("""
                INSERT INTO sessions (session_id, metadata)
                VALUES (?, ?)
                """, (self.session_id, json.dumps({"source": "research_ai"})))
                
                conn.commit()
        
        except Exception as e:
            logger.error(f"Error creating new session: {str(e)}")
