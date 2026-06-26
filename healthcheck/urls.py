from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView

from .views import prometheus_metrics_view

urlpatterns = [
    path("", RedirectView.as_view(pattern_name='pubsub:login'), name='home'),
    path("admin/", admin.site.urls),
    path("i18n/", include("django.conf.urls.i18n")),
    path("metrics", prometheus_metrics_view, name="metrics"),
    path("pubsub/", include("pubsub.urls")),
    path("clinical-ai/", include("clinical_ai.urls")),
]
