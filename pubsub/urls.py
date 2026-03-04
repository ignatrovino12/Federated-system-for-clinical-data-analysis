from django.urls import path
from . import views

app_name = 'pubsub'

urlpatterns = [
    # Authentication
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    
    # Dashboard
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('profile/', views.profile_view, name='profile'),
    
    # Patient Management
    path('patients/', views.patient_list_view, name='patient_list'),
    path('patients/add/', views.patient_add_view, name='patient_add'),
    path('patients/<int:patient_id>/', views.patient_detail_view, name='patient_detail'),
    path('patients/<int:patient_id>/edit/', views.patient_edit_view, name='patient_edit'),
    path('patients/<int:patient_id>/delete/', views.patient_delete_view, name='patient_delete'),
    
    # Appointment Management
    path('appointments/', views.appointment_list_view, name='appointment_list'),
    path('appointments/create/', views.appointment_create_view, name='appointment_create'),
    path('appointments/<int:appointment_id>/', views.appointment_detail_view, name='appointment_detail'),
    path('appointments/<int:appointment_id>/update-status/', views.appointment_update_status_view, name='appointment_update_status'),
    
    # AJAX endpoints
    path('ajax/search-patients/', views.appointment_search_patients_ajax, name='ajax_search_patients'),
]
