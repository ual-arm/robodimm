"""
Authentication and session management.
"""
from fastapi import Header, Depends, HTTPException, Query
from typing import Optional
import uuid
from .models import LoginRequest
from .session import SimulationSession, sessions


# Simple User Store (In production, use a database and hash passwords)
VALID_USERS = {
    "admin": "robotics",
    "user1": "robot1",
    "user2": "robot2",
    "user3": "robot3",
    "user4": "robot4",
    "user5": "robot5",
    "guest": "demo"
}


def login(creds: LoginRequest):
    """Simple login that returns a session ID if credentials are valid."""
    if creds.username in VALID_USERS and VALID_USERS[creds.username] == creds.password:
        # Generate a new session ID (UUID)
        new_session_id = str(uuid.uuid4())
        print(f"[AUTH] Login successful for {creds.username}, creating session {new_session_id}")
        # Pre-create the session
        sessions[new_session_id] = SimulationSession(new_session_id)
        return {"ok": True, "session_id": new_session_id}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")


async def get_session(
    x_session_id: Optional[str] = Header(None),
    session_id: Optional[str] = Query(None)
) -> SimulationSession:
    """Dependency to retrieve the session. ENFORCES AUTHENTICATION."""
    # Prioritize header, then query param
    sid = x_session_id or session_id
    
    if not sid:
        raise HTTPException(status_code=401, detail="Missing session ID")
    
    if sid not in sessions:
        if sid not in sessions:
            raise HTTPException(status_code=401, detail="Invalid or expired session. Please login.")
        
    return sessions[sid]
