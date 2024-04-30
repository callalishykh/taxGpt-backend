from rest_framework import serializers
from .models import Chat, Message


class ChatSerializer(serializers.ModelSerializer):
    class Meta:
        model = Chat
        fields = ["id", "user", "document_file"]
        read_only_fields = ["user"]

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ["id", "chat", "user_prompt", "ai_response", "created_at"]
