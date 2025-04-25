from django.db import models

# Create your models here.
# OK!

class LLMConversation(models.Model):
    """
    The LLMConversation object represents a conversation with a language model.

    Attributes:
        screening: The screening ID associated with the conversation.
        conversation_id: The unique identifier for the conversation.
        conversation: The historical conversation data stored as an array of JSON objects.
            Example:
            [
                {
                    "role": "human",
                    "content": "What is the capital of France?"
                },
                {
                    "role": "ai",
                    "content": "The capital of France is Paris."
                }
            ]
    """

    screening = models.IntegerField()
    conversation_id = models.IntegerField()
    conversation = models.JSONField()
    