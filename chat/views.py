from rest_framework import viewsets
from .models import Chat, Message
from .serializers import ChatSerializer, MessageSerializer
from django.views.decorators.csrf import csrf_exempt
import fitz
from langchain.memory import ConversationKGMemory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.permissions import IsAuthenticated
from langchain_openai import ChatOpenAI
from langchain.chains.conversation.base import ConversationChain
from langchain.chains import TransformChain
from langchain.prompts.prompt import PromptTemplate
import base64
from langchain_core.runnables import chain
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
import json
import os
from django.conf import settings
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from rest_framework.parsers import FormParser, MultiPartParser

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import JSONParser
from django.http import JsonResponse
import os

print(os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY")
memory = ConversationBufferMemory()
llm = ChatOpenAI(
    temperature=0.5, model="gpt-4-turbo", openai_api_key=os.getenv("OPENAI_API_KEY")
)


def extract_text_from_pdf(pdf_path):
    print(pdf_path, "pdf_path")
    # file_path = os.path.join(settings.BASE_DIR, pdf_path)

    doc = fitz.open(pdf_path)

    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def load_image(inputs):

    file_path = os.path.join(settings.BASE_DIR, inputs["image_path"])

    try:
        with open(file_path, "rb") as file:
            print(file_path, "image_path")
            content = file.read()
            base64_encoded = base64.b64encode(content)

            # Decode base64 bytes to string to store in a text field, JSON, etc.
            base64_string = base64_encoded.decode("utf-8")
            print(base64_string, "base64_string")

            return {"image": base64_string}
    except FileNotFoundError:
        print(f"The file {file_path} does not exist in {settings.DOCUMENTS_DIR}.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {str(e)}")
        return None


load_image_chain = TransformChain(
    input_variables=["image_path"], output_variables=["image"], transform=load_image
)


@chain
def image_model(inputs: dict) -> str | list[str] | dict:
    """Invoke model with image and prompt."""
    # we can also use gpt-4-vision-preview but I'm using the gpt-4-turbo as it support images
    # llm = ChatOpenAI(temperature=0.5, model="gpt-4-vision-preview", max_tokens=1024)

    msg = llm.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": inputs["prompt"]},
                    {
                        "type": "text",
                        "text": "Parse this image and retrieve all the detail from this message",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{inputs['image']}"
                        },
                    },
                ]
            )
        ]
    )
    return msg.content


def get_image_information(image_path):
    """This retrieves text from the image using gpt model

    Parameters:
    image_path (string): Path of the image

    """
    vision_prompt = """
    Please parse the given Image and give all details:
    """
    vision_chain = load_image_chain | image_model
    image_info = vision_chain.invoke(
        {"image_path": f"{image_path}", "prompt": vision_prompt}
    )
    return image_info


class ChatViewSet(viewsets.ModelViewSet):
    parser_classes = (MultiPartParser, FormParser)

    queryset = Chat.objects.all()
    serializer_class = ChatSerializer
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Chat.objects.filter(user=self.request.user).order_by("created_at")

    def perform_create(self, serializer):
        chat_data = serializer.validated_data
        serializer.save(user=self.request.user)
        chat_instance = serializer.instance
        document_file = chat_instance.document_file

        content_type = chat_data["document_file"].content_type
        if content_type == "application/pdf":
            initial_message_content = extract_text_from_pdf(document_file)
        else:
            initial_message_content = get_image_information(document_file)

        print(initial_message_content, "initial_message_content")
        self.create_initial_message(chat_instance, initial_message_content)

    def create_initial_message(self, chat, content):
        message = Message(
            chat=chat,
            user_prompt="Please parse the complete file and provide with the details",
            ai_response=content,
        )
        message.save()
        return message


class MessageView(APIView):
    parser_classes = [JSONParser]
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, chat_id, *args, **kwargs):
        if chat_id:
            try:
                conversation = Chat.objects.get(id=chat_id)
                chat_messages = Message.objects.filter(chat=conversation).order_by(
                    "created_at"
                )
                serializer = MessageSerializer(chat_messages, many=True)
                return JsonResponse(serializer.data, safe=False)
            except conversation.DoesNotExist:
                return JsonResponse(
                    {"error": "Conversation not found"},
                    status=status.HTTP_404_NOT_FOUND,
                )
        else:
            return JsonResponse(
                {"error": "Title not provided"}, status=status.HTTP_400_BAD_REQUEST
            )

    def post(self, request, chat_id, *args, **kwargs):
        prompt = request.data.get("prompt")
        user = request.user

        if not chat_id:
            return JsonResponse(
                {"error": "Chat Id is missing"}, status=status.HTTP_400_BAD_REQUEST
            )

        chat = Chat.objects.get(id=chat_id)

        retrieved_chat_history = retrieve_conversation(chat_id)
        reloaded_chain = ConversationChain(
            llm=llm,
            memory=ConversationBufferMemory(chat_memory=retrieved_chat_history),
            verbose=True,
        )
        response = reloaded_chain.predict(input=prompt)
        message = Message.objects.create(
            chat=chat, user_prompt=prompt, ai_response=response
        )

        serializer = MessageSerializer(message)
        return JsonResponse(
            serializer.data,
            status=status.HTTP_201_CREATED,
        )


def retrieve_conversation(chat_id):
    """This retrieves the old conversation of the chat

    @chat_id: id of the chat(number)

    """
    num_recent_conversations = 50

    conversation_context = Message.objects.filter(chat_id=chat_id).order_by(
        "-created_at"
    )[:num_recent_conversations:-1]

    lst = []
    for msg in conversation_context:
        input_msg = getattr(msg, "user_prompt")
        output_msg = getattr(msg, "ai_response")
        lst.append({"input": input_msg, "output": output_msg})

    for x in lst:
        inputs = {"input": x["input"]}
        outputs = {"output": x["output"]}
        memory.save_context(inputs, outputs)

    retrieved_chat_history = ChatMessageHistory(messages=memory.chat_memory.messages)

    return retrieved_chat_history
