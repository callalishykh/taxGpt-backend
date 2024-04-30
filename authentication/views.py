from django.http import JsonResponse
from .serializers import UserSerializer
from rest_framework.parsers import JSONParser
from rest_framework import status
from rest_framework import status
from rest_framework.views import APIView


class RegisterView(APIView):
    parser_classes = [JSONParser]

    def post(self, request, format=None):

        data = JSONParser().parse(request)
        serializer = UserSerializer(data=data)
        valid = serializer.is_valid()
        if valid:
            serializer.save()
            return JsonResponse(
                {"message": "User Registered Successfully"},
                status=status.HTTP_201_CREATED,
            )
        return JsonResponse(
            {"errors": serializer.errors}, status=status.HTTP_400_BAD_REQUEST
        )
