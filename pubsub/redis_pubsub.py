import json
import redis
from django.conf import settings
from datetime import timezone


# Initialize Redis connection
redis_client = redis.Redis(
    host=settings.REDIS_HOST if hasattr(settings, 'REDIS_HOST') else 'redis',
    port=settings.REDIS_PORT if hasattr(settings, 'REDIS_PORT') else 6379,
    db=0,
    decode_responses=True
)


class AppointmentNotifier:
    """Handle Redis pub/sub for appointment notifications"""
    
    CHANNEL_PREFIX = 'appointment'
    
    @staticmethod
    def publish_appointment_created(appointment):
        """
        Publish notification when a new appointment is created
        
        """
        channel = f"{AppointmentNotifier.CHANNEL_PREFIX}:doctor:{appointment.doctor.id}"
        
        message = {
            'type': 'appointment_created',
            'appointment_id': appointment.id,
            'patient_name': appointment.patient.get_full_name(),
            'patient_id': appointment.patient.id,
            'doctor_id': appointment.doctor.id,
            'doctor_name': appointment.doctor.get_full_name() or appointment.doctor.username,
            'appointment_date': appointment.appointment_date.strftime('%Y-%m-%d'),
            'appointment_time': appointment.appointment_time.strftime('%H:%M'),
            'reason': appointment.reason,
            'created_by': appointment.created_by.username if appointment.created_by else 'System',
            'timestamp': appointment.created_at.isoformat()
        }
        
        try:
            # Publish to Redis
            redis_client.publish(channel, json.dumps(message))
            
            # Mark as sent
            appointment.notification_sent = True
            appointment.save(update_fields=['notification_sent'])
            
            return True
        except Exception as e:
            print(f"Error publishing appointment notification: {e}")
            return False
    
    @staticmethod
    def publish_appointment_updated(appointment, action='updated'):
        """
        Publish notification when appointment is updated or cancelled
        Action type (updated, cancelled, confirmed)
        """
        channel = f"{AppointmentNotifier.CHANNEL_PREFIX}:doctor:{appointment.doctor.id}"
        
        message = {
            'type': f'appointment_{action}',
            'appointment_id': appointment.id,
            'patient_name': appointment.patient.get_full_name(),
            'appointment_date': appointment.appointment_date.strftime('%Y-%m-%d'),
            'appointment_time': appointment.appointment_time.strftime('%H:%M'),
            'status': appointment.status,
            'timestamp': appointment.updated_at.isoformat()
        }
        
        try:
            redis_client.publish(channel, json.dumps(message))
            return True
        except Exception as e:
            print(f"Error publishing appointment update: {e}")
            return False
    
    @staticmethod
    def subscribe_to_doctor_appointments(doctor_id, callback):
        """
        Subscribe to appointment notifications for a specific doctor
        """
        channel = f"{AppointmentNotifier.CHANNEL_PREFIX}:doctor:{doctor_id}"
        pubsub = redis_client.pubsub()
        
        try:
            pubsub.subscribe(channel)
            
            for message in pubsub.listen():
                if message['type'] == 'message':
                    data = json.loads(message['data'])
                    callback(data)
        except Exception as e:
            print(f"Error in subscription: {e}")
        finally:
            pubsub.unsubscribe(channel)
            pubsub.close()
    
    @staticmethod
    def get_unread_count(doctor_id):
        """
        Get count of unread appointment notifications for a doctor

        """
        from .models import Appointment
        return Appointment.objects.filter(
            doctor_id=doctor_id,
            notification_sent=True,
            notification_read=False
        ).count()
    
    @staticmethod
    def mark_as_read(appointment_id):
        """
        Mark appointment notification as read
        
        """
        from .models import Appointment
        try:
            appointment = Appointment.objects.get(id=appointment_id)
            appointment.notification_read = True
            appointment.save(update_fields=['notification_read'])
            return True
        except Appointment.DoesNotExist:
            return False


class ClinicBroadcaster:
    """Broadcast system-wide messages to all clinic staff"""
    
    CHANNEL = 'clinic:broadcast'
    
    @staticmethod
    def broadcast_message(message_type, data):
        """
        Broadcast a message to all clinic staff
        
        """
        message = {
            'type': message_type,
            'data': data,
            'timestamp': timezone.now().isoformat()
        }
        
        try:
            redis_client.publish(ClinicBroadcaster.CHANNEL, json.dumps(message))
            return True
        except Exception as e:
            print(f"Error broadcasting message: {e}")
            return False
