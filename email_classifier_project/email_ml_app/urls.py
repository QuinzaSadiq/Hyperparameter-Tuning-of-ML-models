from django.urls import path
from .views import classify_emails

urlpatterns = [
    path('results/', classify_emails, name='classify_emails'),
]
