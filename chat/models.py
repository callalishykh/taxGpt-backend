from django.db import models
from django.contrib.auth.models import User


class Chat(models.Model):
    class Meta:
        app_label = "chat"

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    document_file = models.FileField(upload_to="documents/W-2/")


class Message(models.Model):
    class Meta:
        app_label = "chat"

    chat = models.ForeignKey(Chat, on_delete=models.CASCADE, related_name="messages")
    user_prompt = models.TextField(max_length=3000, default="")
    ai_response = models.TextField(max_length=3000, default="")
    created_at = models.DateTimeField(auto_now_add=True)
