"""
WebSocket server using Flask-SocketIO for real-time communication
"""

import logging
from typing import Optional, Dict, Any, List, Union, Set

from flask import Flask, request
from flask_socketio import SocketIO, emit, join_room, leave_room

from realtime.events import EventDispatcher, EventTypes, Event, CallbackObserver


class SocketIOServer:
    """
    WebSocket server using Flask-SocketIO for real-time dashboard updates
    
    Relays events from the EventDispatcher to connected WebSocket clients.
    """
    
    def __init__(
        self, 
        app: Optional[Flask] = None, 
        event_dispatcher: Optional[EventDispatcher] = None,
        async_mode: str = 'threading', 
        cors_allowed_origins: Union[str, List, None] = '*'
    ):
        """
        Initialize the SocketIO server
        
        Args:
            app: Flask application
            event_dispatcher: Event dispatcher
            async_mode: SocketIO async mode
            cors_allowed_origins: CORS allowed origins
        """
        self.app = app
        self.event_dispatcher = event_dispatcher
        self.socketio = SocketIO(
            app=app, 
            async_mode=async_mode, 
            cors_allowed_origins=cors_allowed_origins,
            logger=True, 
            engineio_logger=True
        )
        
        self.logger = logging.getLogger("realtime.socketio_server")
        self.logger.info("SocketIO server initialized")
        
        # Register event handlers
        self._register_handlers()
        
        # Register observer with event dispatcher to relay events
        if self.event_dispatcher:
            # Add an observer to relay all events
            observer = CallbackObserver(
                callback=self._relay_event_callback,
                name="SocketIORelayObserver"
            )
            self.event_dispatcher.add_observer(observer)
    
    def _register_handlers(self):
        """Register SocketIO event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect(sid=None):
            """Handle new WebSocket connections"""
            self.logger.info(f"Client connected: {request.sid if hasattr(request, 'sid') else 'unknown'}")
            
        @self.socketio.on('disconnect')
        def handle_disconnect(sid=None):
            """Handle WebSocket disconnections"""
            self.logger.info(f"Client disconnected: {request.sid if hasattr(request, 'sid') else 'unknown'}")
            
        @self.socketio.on('subscribe')
        def handle_subscribe(data):
            """
            Handle subscription requests
            
            Expected format:
            {
                'events': ['event_type1', 'event_type2', ...],
                'room': 'optional_room_name'
            }
            """
            client_id = request.sid if hasattr(request, 'sid') else "unknown"
            events = data.get('events', [])
            room = data.get('room')
            
            # Join room if specified
            if room:
                self._join_room(room)
                room_suffix = f" in room {room}"
            else:
                room_suffix = ""
                
            self.logger.debug(f"Client {client_id} subscribed to events {events}{room_suffix}")
            
            # Send history if available
            if self.event_dispatcher and events:
                for event_type in events:
                    history = self.event_dispatcher.get_history(event_type, limit=50)
                    if history and event_type in history and history[event_type]:
                        emit('history', {
                            'event_type': event_type,
                            'events': history[event_type]
                        })
        
        @self.socketio.on('unsubscribe')
        def handle_unsubscribe(data):
            """
            Handle unsubscription requests
            
            Expected format:
            {
                'room': 'room_name'
            }
            """
            client_id = request.sid if hasattr(request, 'sid') else "unknown"
            room = data.get('room')
            
            if room:
                self._leave_room(room)
                self.logger.debug(f"Client {client_id} left room {room}")
                
        @self.socketio.on('admin_command')
        def handle_admin_command(data):
            """
            Handle admin commands
            
            Expected format:
            {
                'command': 'command_name',
                'params': {}
            }
            """
            client_id = request.sid if hasattr(request, 'sid') else "unknown"
            command = data.get('command')
            params = data.get('params', {})
            
            self.logger.debug(f"Client {client_id} sent admin command: {command}")
            
            # Process commands
            result = {
                'success': False,
                'message': 'Unknown command'
            }
            
            if command == 'clear_history' and self.event_dispatcher:
                event_type = params.get('event_type')
                self.event_dispatcher.clear_history(event_type)
                result = {
                    'success': True,
                    'message': f"History cleared for {event_type if event_type else 'all events'}"
                }
            
            # Send result
            emit('command_result', {
                'command': command,
                'result': result
            })
    
    def _relay_event_callback(self, event: Event):
        """Callback to relay an event to all subscribed clients"""
        event_type = event.event_type
        data = event.data.copy()
        
        # Add timestamp if not present
        if 'timestamp' not in data and hasattr(event, 'formatted_time'):
            data['timestamp'] = event.formatted_time
        
        self._relay_event(event_type, data)
    
    def _relay_event(self, event_type: str, data: Dict[str, Any]):
        """
        Relay an event to all subscribed clients
        
        Args:
            event_type: Type of event
            data: Event data
        """
        # Emit to all clients
        self.socketio.emit(event_type, {"data": data})
        
        # Log the relay
        self.logger.debug(f"Relayed event: {event_type}")
    
    def _join_room(self, room: str):
        """
        Join a room
        
        Args:
            room: Room name
        """
        join_room(room)
    
    def _leave_room(self, room: str):
        """
        Leave a room
        
        Args:
            room: Room name
        """
        leave_room(room)
    
    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False, **kwargs):
        """
        Run the SocketIO server
        
        Args:
            host: Host address
            port: Port number
            debug: Debug mode
            **kwargs: Additional parameters for socketio.run()
        """
        self.socketio.run(self.app, host=host, port=port, debug=debug, **kwargs)
    
    def start_background_task(self, target, *args, **kwargs):
        """
        Start a background task
        
        Args:
            target: Function to run
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
        """
        return self.socketio.start_background_task(target, *args, **kwargs)
    
    def stop(self):
        """Stop the SocketIO server"""
        # Not needed when running with Flask
        pass