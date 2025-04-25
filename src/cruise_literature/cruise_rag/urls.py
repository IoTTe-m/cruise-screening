from django.urls import path
from .views import ask_agent

app_name = "cruise_rag"
urlpatterns = [
    path('ask/<int:screening_id>/<int:conversation_id>/', ask_agent, name='ask_agent'),
]
