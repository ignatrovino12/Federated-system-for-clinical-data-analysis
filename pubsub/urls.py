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
]
