from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ChatViewSet, MessageView

app_name = "chat"

router = DefaultRouter()
router.register(r"", ChatViewSet)


urlpatterns = [
    path("conversation/", include(router.urls)),
    path("<int:chat_id>/message/", MessageView.as_view()),
]
