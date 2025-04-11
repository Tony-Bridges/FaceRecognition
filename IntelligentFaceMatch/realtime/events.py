"""
Event system for real-time updates
"""

import logging
import time
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Set, Callable, Union


class EventTypes(str, Enum):
    """Event types for the system"""
    # Face-related events
    FACE_ADDED = "face_added"
    FACE_DELETED = "face_deleted"
    FACE_DETECTED = "face_detected"
    FACE_RECOGNIZED = "face_recognized"
    
    # Camera-related events
    CAMERA_CONNECTED = "camera_connected"
    CAMERA_DISCONNECTED = "camera_disconnected"
    CAMERA_ERROR = "camera_error"
    
    # System events
    SYSTEM_STATUS = "system_status"
    
    # Advanced feature events
    EMOTION_DETECTED = "emotion_detected"
    AGE_GENDER_ESTIMATED = "age_gender_estimated"
    PERSON_TRACKING = "person_tracking"


class Event:
    """Event object with metadata"""
    def __init__(
        self, 
        event_type: Union[str, EventTypes], 
        data: Dict[str, Any],
        timestamp: Optional[float] = None
    ):
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp or time.time()
        self.formatted_time = datetime.fromtimestamp(self.timestamp).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "event_type": self.event_type,
            "data": self.data,
            "timestamp": self.timestamp,
            "formatted_time": self.formatted_time
        }


class EventObserver:
    """Observer for events"""
    def __init__(self, event_types: Optional[List[Union[str, EventTypes]]] = None):
        self.event_types = set(event_types) if event_types else set()
    
    def update(self, event: Event) -> None:
        """Process an event"""
        raise NotImplementedError("Subclasses must implement update()")
    
    def is_interested_in(self, event_type: Union[str, EventTypes]) -> bool:
        """Check if the observer is interested in this event type"""
        return len(self.event_types) == 0 or event_type in self.event_types


class CallbackObserver(EventObserver):
    """Observer that calls a callback function for events"""
    def __init__(
        self, 
        callback: Callable[[Event], None],
        event_types: Optional[List[Union[str, EventTypes]]] = None,
        name: Optional[str] = None
    ):
        super().__init__(event_types)
        self.callback = callback
        self.name = name or f"CallbackObserver-{id(self)}"
    
    def update(self, event: Event) -> None:
        """Process an event by calling the callback"""
        self.callback(event)


class EventDispatcher:
    """Dispatches events to interested observers"""
    def __init__(
        self, 
        max_history_per_type: int = 100,
        history_enabled: bool = True
    ):
        self.observers: List[EventObserver] = []
        self.max_history_per_type = max_history_per_type
        self.history_enabled = history_enabled
        self.history: Dict[str, List[Dict[str, Any]]] = {}
        self.logger = logging.getLogger("realtime.events")
        self.logger.info("Event dispatcher initialized")
    
    def add_observer(self, observer: EventObserver) -> None:
        """Add an observer"""
        self.observers.append(observer)
    
    def remove_observer(self, observer: EventObserver) -> None:
        """Remove an observer"""
        if observer in self.observers:
            self.observers.remove(observer)
    
    def dispatch(
        self, 
        event_type: Union[str, EventTypes], 
        data: Dict[str, Any] = None
    ) -> None:
        """Dispatch an event to all interested observers"""
        event = Event(event_type, data or {})
        
        # Log the event
        self.logger.debug(f"Dispatching event: {event_type}")
        
        # Store in history if enabled
        if self.history_enabled:
            self._add_to_history(event)
        
        # Notify observers
        for observer in self.observers:
            if observer.is_interested_in(event_type):
                try:
                    observer.update(event)
                except Exception as e:
                    self.logger.error(f"Error in observer {observer}: {e}")
    
    def _add_to_history(self, event: Event) -> None:
        """Add an event to the history"""
        event_type = event.event_type
        
        # Initialize history for this type if it doesn't exist
        if event_type not in self.history:
            self.history[event_type] = []
        
        # Add event to history
        event_dict = event.data.copy()
        event_dict["timestamp"] = event.formatted_time
        self.history[event_type].append(event_dict)
        
        # Trim history if necessary
        if len(self.history[event_type]) > self.max_history_per_type:
            self.history[event_type] = self.history[event_type][-self.max_history_per_type:]
    
    def get_history(
        self, 
        event_type: Optional[Union[str, EventTypes]] = None,
        limit: Optional[int] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get event history"""
        if not self.history_enabled:
            return {}
            
        if event_type:
            # Return history for a specific event type
            events = self.history.get(event_type, [])
            if limit and limit > 0:
                events = events[-limit:]
            return {event_type: events}
        else:
            # Return all history
            result = {}
            for event_type, events in self.history.items():
                if limit and limit > 0:
                    result[event_type] = events[-limit:]
                else:
                    result[event_type] = events
            return result
    
    def clear_history(self, event_type: Optional[Union[str, EventTypes]] = None) -> None:
        """Clear event history"""
        if event_type:
            # Clear history for a specific event type
            if event_type in self.history:
                self.history[event_type] = []
        else:
            # Clear all history
            self.history = {}


# Singleton instance
_event_dispatcher = None

def get_event_dispatcher() -> EventDispatcher:
    """Get the singleton event dispatcher"""
    global _event_dispatcher
    if _event_dispatcher is None:
        _event_dispatcher = EventDispatcher()
    return _event_dispatcher